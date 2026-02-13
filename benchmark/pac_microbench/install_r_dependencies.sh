#!/bin/bash
# Install system dependencies for R packages needed for plotting with patterns

echo "=== Installing system dependencies for R packages ==="

# Check if we have sudo access
if ! sudo -n true 2>/dev/null; then
    echo "Note: This script requires sudo access to install system packages."
    echo "You may be prompted for your password."
fi

# Install system dependencies
echo "Installing udunits2-devel (for units package)..."
sudo dnf install -y udunits2-devel

echo "Installing GDAL dependencies (for sf package)..."
sudo dnf install -y gdal-devel proj-devel geos-devel sqlite-devel

echo "Installing other graphics dependencies..."
sudo dnf install -y libpng-devel libjpeg-turbo-devel libtiff-devel

echo ""
echo "=== Installing R packages ==="

# Install R packages in the correct order
Rscript -e '
user_lib <- Sys.getenv("R_LIBS_USER")
if (user_lib == "") {
  user_lib <- file.path(Sys.getenv("HOME"), "R", "libs")
}
if (!dir.exists(user_lib)) {
  dir.create(user_lib, recursive = TRUE, showWarnings = FALSE)
}
.libPaths(c(user_lib, .libPaths()))

options(repos = c(CRAN = "https://cloud.r-project.org"))

# Install packages in order
packages <- c("units", "sf", "gridpattern", "ggpattern")

for (pkg in packages) {
  cat("\n=== Installing", pkg, "===\n")
  if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
    install.packages(pkg, dependencies = TRUE, lib = user_lib)
  } else {
    cat(pkg, "is already installed\n")
  }
}

cat("\n=== Verifying installation ===\n")
for (pkg in packages) {
  if (require(pkg, character.only = TRUE, quietly = TRUE)) {
    cat("✓", pkg, "loaded successfully\n")
  } else {
    cat("✗", pkg, "failed to load\n")
  }
}
'

echo ""
echo "=== Installation complete ==="

