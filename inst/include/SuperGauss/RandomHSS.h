/// @file RandomHSS.h

#ifndef RandomHSS_h
#define RandomHSS_h 1

#include "ComplexFFT.h"
#include <complex>
#include <cstring> // for std::memcpy

using std::complex<double> dcomplex;

/// @brief Storage for Toeplitz matrix and its Cauchy-like representation.
class MatrixWrapper {
private:
  int N_; ///< Size of Toeplitz matrix.

public:
  dcomplex *cv_; ///< First column of matrix.
  dcomplex *rv_; ///< First row of matrix.
  dcomplex *d_;
  dcomplex *x_;
  dcomplex *u_;
  dcomplex *v_;

  /// Constructor.
  MatrixWrapper(int N) : N_(N) {
    int N2 = 2 * N;
    new dcomplex *cv_[N];
    new dcomplex *rv_[N];
    new dcomplex *d_[N];
    new dcomplex *x_[N];
    new dcomplex *u_[N2];
    new dcomplex *v_[N2];
  }

  /// Destructor.
  ~MatrixWrapper() {
    delete[] cv_;
    delete[] rv_;
    delete[] d_;
    delete[] x_;
    delete[] u_;
    delete[] v_;
  }
};

/// @brief Functor to perform Toeplitz matrix-matrix multiplication using FFT.
///
/// This class performs multiplication y = T * x where T is a Toeplitz matrix
/// with first column `cv` and first row `rv`, and `x` is a matrix with `ncols`
/// columns. Each column of `x` is multiplied individually, using FFT-based
/// convolution. FFTs are computed using an external `ComplexFFT` object
/// provided at construction.
///
/// This functor gets called multiple times, but always with the same size.
class ToepMV {
private:
  int N_;          ///< Problem size (length of input/output vectors)
  dcomplex *c_;    ///< Toeplitz convolution kernel
  dcomplex *c1_;   ///< FFT of Toeplitz kernel
  dcomplex *xt_;   ///< Extended input vector
  dcomplex *xtt_;  ///< Temporary vector for FFT results
  ComplexFFT fft_; ///< Complex FFTs of size 2N.

public:
  /// @brief Construct the Toeplitz matrix-vector multiplication functor.
  ///
  /// @param N Size of the input vectors (number of rows).
  ToepMV(int N) : N_(N), fft_(N) {
    int N2 = 2 * N_;
    c_ = new dcomplex[N2];
    c1_ = new dcomplex[N2];
    xt_ = new dcomplex[N2];
    xtt_ = new dcomplex[N2];
  }

  /// @brief Destructor to deallocate internal buffers.
  ~ToepMV() {
    delete[] c_;
    delete[] c1_;
    delete[] xt_;
    delete[] xtt_;
  }

  /// @brief Perform Toeplitz matrix-matrix multiplication.
  ///
  /// Computes y = T * x for a Toeplitz matrix T with first column `cv` and
  /// first row `rv`. The input `x` is a flattened column-major matrix of shape
  /// `n × ncols`, and `y` is an output matrix of the same shape.
  ///
  /// @param[out] y Output array of size `N × ncols`.
  /// @param[in]  x Input array of size `N × ncols`.
  /// @param[in]  cv First column of the Toeplitz matrix (size n).
  /// @param[in]  rv First row of the Toeplitz matrix (size n).
  /// @param[in]  ncols Number of columns in the input/output matrix.
  ///
  /// @todo
  /// - Reuse with same input kernel, i.e., compute `c_` and `c1_` only once.
  void operator()(dcomplex *y, const dcomplex *x, const dcomplex *cv,
                  const dcomplex *rv, int ncols) {
    int N = N_;
    int N2 = 2 * N;
    double scale = 1.0 / double(N2);

    // Form Toeplitz kernel: c = [cv; 0; reverse(rv[1:])]
    std::memcpy(c_, cv, N * sizeof(dcomplex));
    c_[N] = dcomplex(0.0, 0.0);
    for (int i = 1; i < N; ++i)
      c_[N + i] = rv[N - i];

    // Compute iFFT of kernel
    fft_.ifft(c1_, c_);

    for (int col = 0; col < ncols; ++col) {
      // Grab current i/o columns
      const dcomplex *x_col = x + col * N;
      dcomplex *y_col = y + col * N;

      // Extend input vector: xt = [x_col; 0s]
      std::memcpy(xt_, x_col, N * sizeof(dcomplex));
      std::memset(xt_ + N, 0, N * sizeof(dcomplex));

      // Compute iFFT of input column
      fft_.ifft(xtt_, xt_);

      // Pointwise multiplication
      for (int i = 0; i < N2; ++i) {
        xtt_[i] *= c1_[i];
      }

      // Extract and scale result
      for (int i = 0; i < N; ++i) {
        y_col[i] = xt_[i] * scale;
      }
    }
  }
};

/// @brief Matrix multiplication of the Cauchy-like matrix corresponding to a
/// given Toeplitz matrix `T`.
///
/// This functor gets called twice.
class FMatMulHSSRS {
private:
  int N_;             ///< Size of Toeplitz matrix.
  dcomplex *yt_;      ///< Storage for Toeplitz-vector multiplication.
  dcomplex *cv_conj_; ///< Storage for complex conjugate of first column of `T`.
  dcomplex *rv_conj_; ///< Storage for complex conjugate of ofirst row of `T`.
  ComplexFFT fft_;    ///< Complex FFTs of size N.

  /// Constructor.
  FMatMulHSSRS(int N) : N_(N), fft_(N) {
    yt_ = new dcomplex[N_];
    cv_conj_ = new dcomplex[N_];
    rv_conj_ = new dcomplex[N_];
  }

  /// Destructor.
  ~FMatMulHSSRS() {
    delete[] yt_;
    delete[] cv_conj_;
    delete[] rv_conj_;
  }

  void operator()(dcomplex *y, const dcomplex *x, const MatrixWrapper &A,
                  char trans, int ncols, ToepMV &toepmv) {

    double scale = 1.0 / double(N_);
    for (int col = 0; col < ncols; ++col) {
      // Grab current i/o columns
      const dcomplex *x_col = x + col * N;
      dcomplex *y_col = y + col * N;
      // transform input column
      fft_.fft(y_col, x_col);
      if (trans == "N" || trans == "n") {
        // FIXME: avoid recomputing kernel and its fft each time
        toepmv(yt_, y_col, A.cv, A.rv, 1);
      } else {
        for (int i = 0; i < N_; ++i) {
          cv_conj_[i] = std::conj(A.cv[i]);
          rv_conj_[i] = std::conj(A.rv[i]);
        }
        toepmv(yt_, y_col, cv_conj_, rv_conj_, 1);
      }
      fft_.ifft(y_col, y_t);
      // scale result
      for (int i = 0; i < N_; ++i) {
        y_col[i] *= scale;
      }
    }
  }
};

/// @brief Functor that converts a Toeplitz matrix to its Cauchy-like
/// representation.
///
/// This functor transforms a Toeplitz matrix defined by its first row and
/// column into a Cauchy-like matrix with generators `u`, `v` and diagonal `d`,
/// and modified diagonal `x`. All memory allocations are performed in the
/// constructor and freed in the destructor. Assumes matrices are stored in
/// column-major format.
class Toep2Cauchy {
private:
  int N_;        ///< Size of Toeplitz matrix.
  dcomplex *uu_; ///< IFFT workspace
  dcomplex *vv_; ///< FFT workspace
  dcomplex *x_;  ///< Workspace for x
  dcomplex *d_;  ///< Workspace for d
  ComplexFFT fft_;

  /// @brief Transpose a matrix stored in column-major format.
  /// @param[out] out Output buffer (ncols x nrows).
  /// @param[in] in Input buffer (nrows x ncols).
  /// @param[in] nrows Number of rows in the input matrix.
  /// @param[in] ncols Number of columns in the input matrix.
  void transpose(dcomplex *out, const dcomplex *in, int nrows,
                 int ncols) const {
    for (int i = 0; i < nrows; ++i) {
      for (int j = 0; j < ncols; ++j) {
        out[j + i * ncols] = in[i + j * nrows];
      }
    }
  }

public:
  /// @brief Constructor: allocates internal memory.
  ///
  /// @param N Size of the Toeplitz matrix.
  Toep2Cauchy(int N) : N_(N), fft_(N) {
    uu_ = new dcomplex[2 * N_];
    vv_ = new dcomplex[2 * N_];
    x_ = new dcomplex[N_];
    d_ = new dcomplex[N_];
  }

  /// @brief Destructor.
  ~Toep2Cauchy() {
    delete[] uu_;
    delete[] vv_;
    delete[] x_;
    delete[] d_;
  }

  /// Operator to perform Toeplitz to Cauchy-like transformation
  /// @param A MatrixWrapper object to store the resulting Cauchy-like
  /// components.
  void operator()(MatrixWrapper &A) const {
    const int N = N_;
    const int ncols = 2;

    // Initialize A.u and A.v
    std::memset(A.u, 0, N * ncols * sizeof(dcomplex));
    std::memset(A.v, 0, ncols * N * sizeof(dcomplex));
    A.u[0] = ONE_;         // A.u(1,1)
    A.v[N - 1 + N] = ONE_; // A.v(2,A%n)

    // Fill A.u(:,2)
    for (int i = 1; i < N; ++i)
      A.u[i + N] = A.rv[N + 1 - i] - A.cv[i];

    // Fill A.v(1,1:N-1)
    for (int i = 0; i < N - 1; ++i)
      A.v[i] = -A.u[N - 1 - i + N];

    // Apply IFFT to A.u
    fft_.ifft(A.u, uu, N, ncols);

    for (int i = 0; i < N * ncols; ++i)
      A.u[i] = uu[i] / static_cast<double>(N);

    // Apply FFT to transpose of A.v
    transpose(A.v, vt, ncols, N);
    fft_.fft(vt, vv, N, ncols);
    transpose(vv, A.v, N, ncols);

    // Compute diagonal scaling vector d
    d[0] = ONE_;
    dcomplex omega(std::cos(2.0 * PI_ / N), std::sin(2.0 * PI_ / N));
    for (int i = 1; i < N; ++i)
      d[i] = omega * d[i - 1];

    std::memcpy(A.d, d, N * sizeof(dcomplex));

    // Construct vector x
    x[0] = A.cv[0];
    for (int i = 1; i < N; ++i) {
      x[i] = static_cast<double>(i) * A.rv[N - i] +
             static_cast<double>(N - i) * A.cv[i];
      x[i] /= static_cast<double>(N);
    }

    // Apply FFT to x
    fft_.fft(x, A.x, N, 1);

    // Reverse copy
    dcomplex *xtemp = new dcomplex[N];
    std::memcpy(xtemp, A.x, N * sizeof(dcomplex));
    for (int i = 1; i < N; ++i)
      A.x[i] = xtemp[N - i];
    delete[] xtemp;

    delete[] uu;
    delete[] vv;
    delete[] vt;
    delete[] x;
    delete[] d;
  }
};

#endif
