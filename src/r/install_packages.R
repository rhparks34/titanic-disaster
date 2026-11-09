# Installs the package versions used by titanic_model.R
# remotes comes from the base image's dated CRAN snapshot (2025-10-30), so its version is fixed too
install.packages("remotes")
remotes::install_version("dplyr", version = "1.1.4", upgrade = "never")

# Stop the Docker build if dplyr didn't install
stopifnot(packageVersion("dplyr") == "1.1.4")
