// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppEigen.h>
#include <Rcpp.h>

using namespace Rcpp;

// DurbinLevinson_XZ
Eigen::MatrixXd DurbinLevinson_XZ(Eigen::MatrixXd X, Eigen::VectorXd acf);
RcppExport SEXP SuperGauss_DurbinLevinson_XZ(SEXP XSEXP, SEXP acfSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Eigen::MatrixXd >::type X(XSEXP);
    Rcpp::traits::input_parameter< Eigen::VectorXd >::type acf(acfSEXP);
    rcpp_result_gen = Rcpp::wrap(DurbinLevinson_XZ(X, acf));
    return rcpp_result_gen;
END_RCPP
}
// DurbinLevinson_ZX
Eigen::MatrixXd DurbinLevinson_ZX(Eigen::MatrixXd Z, Eigen::VectorXd acf);
RcppExport SEXP SuperGauss_DurbinLevinson_ZX(SEXP ZSEXP, SEXP acfSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Eigen::MatrixXd >::type Z(ZSEXP);
    Rcpp::traits::input_parameter< Eigen::VectorXd >::type acf(acfSEXP);
    rcpp_result_gen = Rcpp::wrap(DurbinLevinson_ZX(Z, acf));
    return rcpp_result_gen;
END_RCPP
}
// DurbinLevinson_Eigen
Rcpp::List DurbinLevinson_Eigen(Eigen::MatrixXd X, Eigen::MatrixXd Y, Eigen::VectorXd acf, int calcMode);
RcppExport SEXP SuperGauss_DurbinLevinson_Eigen(SEXP XSEXP, SEXP YSEXP, SEXP acfSEXP, SEXP calcModeSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Eigen::MatrixXd >::type X(XSEXP);
    Rcpp::traits::input_parameter< Eigen::MatrixXd >::type Y(YSEXP);
    Rcpp::traits::input_parameter< Eigen::VectorXd >::type acf(acfSEXP);
    Rcpp::traits::input_parameter< int >::type calcMode(calcModeSEXP);
    rcpp_result_gen = Rcpp::wrap(DurbinLevinson_Eigen(X, Y, acf, calcMode));
    return rcpp_result_gen;
END_RCPP
}
// DurbinLevinson_Base
Rcpp::List DurbinLevinson_Base(NumericMatrix X, NumericMatrix Y, NumericVector acf, int calcMode);
RcppExport SEXP SuperGauss_DurbinLevinson_Base(SEXP XSEXP, SEXP YSEXP, SEXP acfSEXP, SEXP calcModeSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericMatrix >::type X(XSEXP);
    Rcpp::traits::input_parameter< NumericMatrix >::type Y(YSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type acf(acfSEXP);
    Rcpp::traits::input_parameter< int >::type calcMode(calcModeSEXP);
    rcpp_result_gen = Rcpp::wrap(DurbinLevinson_Base(X, Y, acf, calcMode));
    return rcpp_result_gen;
END_RCPP
}
// Toeplitz_constructor
SEXP Toeplitz_constructor(int n);
RcppExport SEXP SuperGauss_Toeplitz_constructor(SEXP nSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< int >::type n(nSEXP);
    rcpp_result_gen = Rcpp::wrap(Toeplitz_constructor(n));
    return rcpp_result_gen;
END_RCPP
}
// Toeplitz_setAcf
void Toeplitz_setAcf(SEXP Toep_ptr, NumericVector acf);
RcppExport SEXP SuperGauss_Toeplitz_setAcf(SEXP Toep_ptrSEXP, SEXP acfSEXP) {
BEGIN_RCPP
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type Toep_ptr(Toep_ptrSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type acf(acfSEXP);
    Toeplitz_setAcf(Toep_ptr, acf);
    return R_NilValue;
END_RCPP
}
// Toeplitz_getAcf
NumericVector Toeplitz_getAcf(SEXP Toep_ptr);
RcppExport SEXP SuperGauss_Toeplitz_getAcf(SEXP Toep_ptrSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type Toep_ptr(Toep_ptrSEXP);
    rcpp_result_gen = Rcpp::wrap(Toeplitz_getAcf(Toep_ptr));
    return rcpp_result_gen;
END_RCPP
}
// Toeplitz_getPhi
NumericVector Toeplitz_getPhi(SEXP Toep_ptr);
RcppExport SEXP SuperGauss_Toeplitz_getPhi(SEXP Toep_ptrSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type Toep_ptr(Toep_ptrSEXP);
    rcpp_result_gen = Rcpp::wrap(Toeplitz_getPhi(Toep_ptr));
    return rcpp_result_gen;
END_RCPP
}
// Toeplitz_Multiply
NumericMatrix Toeplitz_Multiply(SEXP Toep_ptr, NumericMatrix X);
RcppExport SEXP SuperGauss_Toeplitz_Multiply(SEXP Toep_ptrSEXP, SEXP XSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type Toep_ptr(Toep_ptrSEXP);
    Rcpp::traits::input_parameter< NumericMatrix >::type X(XSEXP);
    rcpp_result_gen = Rcpp::wrap(Toeplitz_Multiply(Toep_ptr, X));
    return rcpp_result_gen;
END_RCPP
}
// Toeplitz_Solve
NumericMatrix Toeplitz_Solve(SEXP Toep_ptr, NumericMatrix X);
RcppExport SEXP SuperGauss_Toeplitz_Solve(SEXP Toep_ptrSEXP, SEXP XSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type Toep_ptr(Toep_ptrSEXP);
    Rcpp::traits::input_parameter< NumericMatrix >::type X(XSEXP);
    rcpp_result_gen = Rcpp::wrap(Toeplitz_Solve(Toep_ptr, X));
    return rcpp_result_gen;
END_RCPP
}
// Toeplitz_Determinant
double Toeplitz_Determinant(SEXP Toep_ptr);
RcppExport SEXP SuperGauss_Toeplitz_Determinant(SEXP Toep_ptrSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type Toep_ptr(Toep_ptrSEXP);
    rcpp_result_gen = Rcpp::wrap(Toeplitz_Determinant(Toep_ptr));
    return rcpp_result_gen;
END_RCPP
}
// Toeplitz_traceT2
double Toeplitz_traceT2(SEXP Toep_ptr, NumericVector acf2);
RcppExport SEXP SuperGauss_Toeplitz_traceT2(SEXP Toep_ptrSEXP, SEXP acf2SEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type Toep_ptr(Toep_ptrSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type acf2(acf2SEXP);
    rcpp_result_gen = Rcpp::wrap(Toeplitz_traceT2(Toep_ptr, acf2));
    return rcpp_result_gen;
END_RCPP
}
// Toeplitz_traceT4
double Toeplitz_traceT4(SEXP Toep_ptr, NumericVector acf2, NumericVector acf3);
RcppExport SEXP SuperGauss_Toeplitz_traceT4(SEXP Toep_ptrSEXP, SEXP acf2SEXP, SEXP acf3SEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type Toep_ptr(Toep_ptrSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type acf2(acf2SEXP);
    Rcpp::traits::input_parameter< NumericVector >::type acf3(acf3SEXP);
    rcpp_result_gen = Rcpp::wrap(Toeplitz_traceT4(Toep_ptr, acf2, acf3));
    return rcpp_result_gen;
END_RCPP
}
// Toeplitz_hasAcf
bool Toeplitz_hasAcf(SEXP Toep_ptr);
RcppExport SEXP SuperGauss_Toeplitz_hasAcf(SEXP Toep_ptrSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type Toep_ptr(Toep_ptrSEXP);
    rcpp_result_gen = Rcpp::wrap(Toeplitz_hasAcf(Toep_ptr));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"SuperGauss_DurbinLevinson_XZ", (DL_FUNC) &SuperGauss_DurbinLevinson_XZ, 2},
    {"SuperGauss_DurbinLevinson_ZX", (DL_FUNC) &SuperGauss_DurbinLevinson_ZX, 2},
    {"SuperGauss_DurbinLevinson_Eigen", (DL_FUNC) &SuperGauss_DurbinLevinson_Eigen, 4},
    {"SuperGauss_DurbinLevinson_Base", (DL_FUNC) &SuperGauss_DurbinLevinson_Base, 4},
    {"SuperGauss_Toeplitz_constructor", (DL_FUNC) &SuperGauss_Toeplitz_constructor, 1},
    {"SuperGauss_Toeplitz_setAcf", (DL_FUNC) &SuperGauss_Toeplitz_setAcf, 2},
    {"SuperGauss_Toeplitz_getAcf", (DL_FUNC) &SuperGauss_Toeplitz_getAcf, 1},
    {"SuperGauss_Toeplitz_getPhi", (DL_FUNC) &SuperGauss_Toeplitz_getPhi, 1},
    {"SuperGauss_Toeplitz_Multiply", (DL_FUNC) &SuperGauss_Toeplitz_Multiply, 2},
    {"SuperGauss_Toeplitz_Solve", (DL_FUNC) &SuperGauss_Toeplitz_Solve, 2},
    {"SuperGauss_Toeplitz_Determinant", (DL_FUNC) &SuperGauss_Toeplitz_Determinant, 1},
    {"SuperGauss_Toeplitz_traceT2", (DL_FUNC) &SuperGauss_Toeplitz_traceT2, 2},
    {"SuperGauss_Toeplitz_traceT4", (DL_FUNC) &SuperGauss_Toeplitz_traceT4, 3},
    {"SuperGauss_Toeplitz_hasAcf", (DL_FUNC) &SuperGauss_Toeplitz_hasAcf, 1},
    {NULL, NULL, 0}
};

RcppExport void R_init_SuperGauss(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
