source("SuperGauss-testfunctions.R")

context("Circulant - Product.")

test_that("Circulant$prod works as expected.", {
  skip_on_cran()
  case_par <- expand.grid(N = c(1, sample(2:20, 10)),
                          p = 1:3)
  ncase <- nrow(case_par)
  for(ii in 1:ncase) {
    list2env(case_par[ii,], envir = environment())
    Nu <- floor(N/2)+1
    upsd <- exp(-runif(1) * (1:Nu)/Nu)
    acf <- ifft(unfold_acf(N, uacf = upsd))
    X <- matrix(rnorm(N*p), N, p)
    dropX <- (p == 1) && (runif(1) < .5)
    if(dropX) X <- drop(X)
    Y <- circulant(acf) %*% X
    Ct <- Circulant$new(N, uacf = acf[1:Nu])
    expect_equal(Y, Ct$prod(X))
  }
})

