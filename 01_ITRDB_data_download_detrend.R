
# Download data [.rwl] -----
library(RCurl)
library(stringr)

regions <- c("africa", "asia", "atlantic", "australia", "centralamerica", "europe", 
             "northamerica/canada", "northamerica/mexico", "northamerica/usa",
             "southamerica")
base_url <- "ftp://ftp.ncei.noaa.gov/pub/data/paleo/treering/measurements/"


file_path <- "F:/Global_ITRDB_RWI/rwl_data-260128"
dir.create(file_path, showWarnings = FALSE)

# Create a failure log file (clear if exists)
failed_log <- "F:/Global_ITRDB_RWI/download_failed_list_rwl-260128.txt"
if (file.exists(failed_log)) file.remove(failed_log)

for (region in regions) {
  ftp_region_url <- paste0(base_url, region, "/")
  message("📂 Reading region: ", ftp_region_url)
  
  if(region %in% c("northamerica/canada", "northamerica/mexico", "northamerica/usa")){
    region_name <- "northamerica"
  }else{
    region_name <- region
  }
  
  # Get all filenames in the directory (FTP)
  tryCatch({
    file_list <- getURL(ftp_region_url, dirlistonly = TRUE)
    files <- unlist(strsplit(file_list, "\r\n"))
    rwl_files <- files[str_detect(files, "\\.rwl$")]
    rwl_files <- files[
      str_detect(files, "\\.rwl$") & !str_detect(files, "-noaa\\.rwl$")
    ]
    cat(region, length(rwl_files))
    
    # Iterate and download .rwl files
    for (file in rwl_files) {
      dest_path <- file.path(file_path, paste0(region_name, "_", file))
      
      if (file.exists(dest_path)) {
        message("✅ Exists, skipping: ", dest_path)
        next
      }
      
      file_url <- paste0(ftp_region_url, file)
      message("⬇️ Downloading: ", file_url)
      
      tryCatch({
        download.file(file_url, destfile = dest_path, mode = "wb")
      }, error = function(e) {
        message("❌ Download failed: ", file_url)
        write(file_url, file = failed_log, append = TRUE)
      })
    }
  }, error = function(e) {
    message("❌ Cannot access region directory: ", ftp_region_url)
    write(ftp_region_url, file = failed_log, append = TRUE)
  })
}

# Download data [.txt] -----

library(RCurl)
library(stringr)

regions <- c("africa", "asia", "atlantic", "australia", "centralamerica", "europe", 
             "northamerica/canada", "northamerica/mexico", "northamerica/usa",
             "southamerica")
base_url <- "ftp://ftp.ncei.noaa.gov/pub/data/paleo/treering/measurements/"


file_path <- "F:/Global_ITRDB_RWI/txt_data"
dir.create(file_path, showWarnings = FALSE)

# Create a failure log file (clear if exists)
failed_log <- "F:/Global_ITRDB_RWI/download_failed_list_txt.txt"
if (file.exists(failed_log)) file.remove(failed_log)

for (region in regions) {
  ftp_region_url <- paste0(base_url, region, "/")
  message("📂 Reading region: ", ftp_region_url)
  
  if(region %in% c("northamerica/canada", "northamerica/mexico", "northamerica/usa")){
    region_name <- "northamerica"
  }else{
    region_name <- region
  }
  
  # Get all filenames in the directory (FTP)
  tryCatch({
    file_list <- getURL(ftp_region_url, dirlistonly = TRUE)
    files <- unlist(strsplit(file_list, "\r\n"))
    rwl_files <- files[str_detect(files, "\\.txt$")]
    cat(region, length(rwl_files))
    
    # Iterate and download .rwl files (variable name reused from previous block logic)
    for (file in rwl_files) {
      dest_path <- file.path(file_path, paste0(region_name, "_", file))
      
      if (file.exists(dest_path)) {
        message("✅ Exists, skipping: ", dest_path)
        next
      }
      
      file_url <- paste0(ftp_region_url, file)
      message("⬇️ Downloading: ", file_url)
      
      tryCatch({
        download.file(file_url, destfile = dest_path, mode = "wb")
      }, error = function(e) {
        message("❌ Download failed: ", file_url)
        write(file_url, file = failed_log, append = TRUE)
      })
    }
  }, error = function(e) {
    message("❌ Cannot access region directory: ", ftp_region_url)
    write(ftp_region_url, file = failed_log, append = TRUE)
  })
}

# Move files [.rwl] -----
# Set source and target directories
source_dir <- "F:/Global_ITRDB_RWI/rwl_data"
target_dir <- file.path(source_dir, "noaa_rwl_only")

# Create target directory (if not exists)
dir.create(target_dir, showWarnings = FALSE)

# Get all files
all_files <- list.files(source_dir, pattern = "-noaa\\.rwl$", full.names = TRUE)

# Execute file move
for (file_path in all_files) {
  file_name <- basename(file_path)
  dest_path <- file.path(target_dir, file_name)
  
  # Move file (equivalent to cut)
  file.rename(from = file_path, to = dest_path)
  message("📦 Moving file: ", file_name)
}



# Move files [.txt]-----
# Set directories
source_dir <- "F:/Global_ITRDB_RWI/txt_data"
target_dir <- file.path(source_dir, "noaa_txt_only")

# Create target directory
dir.create(target_dir, showWarnings = FALSE)

# Get files matching criteria (ends with -noaa.txt but excludes -rwl-noaa.txt)
all_txt_files <- list.files(source_dir, pattern = "-noaa\\.txt$", full.names = TRUE)
selected_files <- all_txt_files[!grepl("-rwl-noaa\\.txt$", all_txt_files)]

# Execute move operation
for (file_path in selected_files) {
  file_name <- basename(file_path)
  dest_path <- file.path(target_dir, file_name)
  
  file.rename(from = file_path, to = dest_path)
  message("📦 Moved file: ", file_name)
}


# Check if all [.rwl] are downloaded -----
library(RCurl)
library(stringr)
library(dplyr)

# Set regions and paths
regions <- c("africa", "asia", "atlantic", "australia", "centralamerica", "europe", 
             "northamerica/canada", "northamerica/mexico", "northamerica/usa",
             "southamerica")
base_url <- "ftp://ftp.ncei.noaa.gov/pub/data/paleo/treering/measurements/"
file_path <- "F:/Global_ITRDB_RWI/rwl_data"

# Initialize record table
result <- data.frame(
  region = character(),
  region_name = character(),
  total_online = numeric(),
  total_local = numeric(),
  complete = logical(),
  stringsAsFactors = FALSE
)

region <- "europe"
for (region in regions) {
  ftp_region_url <- paste0(base_url, region, "/")
  if(region %in% c("northamerica/canada", "northamerica/mexico", "northamerica/usa")){
    region_name <- "northamerica"
  } else {
    region_name <- region
  }
  
  # Get remote file list
  tryCatch({
    file_list <- getURL(ftp_region_url, dirlistonly = TRUE)
    print(region_name)
    files <- unlist(strsplit(file_list, "\r\n"))
    rwl_files <- files[str_detect(files, "\\.rwl$") & !str_detect(files, "-noaa\\.rwl$")]
    
    # Local file count (prefixed with region_name_)
    local_files <- list.files(file_path, pattern = paste0("^", region_name, "_.*\\.rwl$"))
    
    # Write to result table
    result <- result %>% add_row(
      region = region,
      region_name = region_name,
      total_online = length(rwl_files),
      total_local = length(local_files),
      complete = length(rwl_files) == length(local_files)
    )
  }, error = function(e) {
    message("❌ Cannot read: ", region)
    result <- result %>% add_row(
      region = region,
      region_name = region_name,
      total_online = NA,
      total_local = NA,
      complete = FALSE
    )
  })
}

# View and save results
print(result)
write.csv(result, "F:/Global_ITRDB_RWI/rwl_download_check_summary.csv", row.names = FALSE)

# Check if all [.txt] are downloaded -----
library(RCurl)
library(stringr)
library(dplyr)

# Set regions and paths
regions <- c("africa", "asia", "atlantic", "australia", "centralamerica", "europe", 
             "northamerica/canada", "northamerica/mexico", "northamerica/usa",
             "southamerica")
base_url <- "ftp://ftp.ncei.noaa.gov/pub/data/paleo/treering/measurements/"
file_path <- "F:/Global_ITRDB_RWI/txt_data"

# Initialize record table
result <- data.frame(
  region = character(),
  region_name = character(),
  total_online = numeric(),
  total_local = numeric(),
  complete = logical(),
  stringsAsFactors = FALSE
)

region <- "southamerica"
for (region in regions) {
  ftp_region_url <- paste0(base_url, region, "/")
  if(region %in% c("northamerica/canada", "northamerica/mexico", "northamerica/usa")){
    region_name <- "northamerica"
  } else {
    region_name <- region
  }
  
  # Get remote file list
  tryCatch({
    file_list <- getURL(ftp_region_url, dirlistonly = TRUE)
    print(region_name)
    files <- unlist(strsplit(file_list, "\r\n"))
    txt_files <- files[grepl("-rwl-noaa\\.txt$", files)]
    
    # Local file count (prefixed with region_name_)
    local_files <- list.files(file_path, pattern = paste0("^", region_name, "_.*\\.txt$"))
    
    # Write to result table
    result <- result %>% add_row(
      region = region,
      region_name = region_name,
      total_online = length(txt_files),
      total_local = length(local_files),
      complete = length(txt_files) == length(local_files)
    )
  }, error = function(e) {
    message("❌ Cannot read: ", region)
    result <- result %>% add_row(
      region = region,
      region_name = region_name,
      total_online = NA,
      total_local = NA,
      complete = FALSE
    )
  })
}

# View and save results
print(result)
write.csv(result, "F:/Global_ITRDB_RWI/txt_download_check_summary.csv", row.names = FALSE)

# Calculate detrended chronologies -----
library(dplR)
library(stringr)

# Set paths
input_dir <- "F:/Global_ITRDB_RWI/rwl_data"
output_dir <- "F:/Global_ITRDB_RWI/chron_csv"
error_log_path <- "F:/Global_ITRDB_RWI/chron_errors.txt"
summary_csv <- "F:/Global_ITRDB_RWI/chron_statistics_summary.csv"

dir.create(output_dir, showWarnings = FALSE)
if (file.exists(error_log_path)) file.remove(error_log_path)

# Initialize statistics summary
summary_list <- list()
error_count <- 0

# Iterate through all .rwl files
rwl_files <- list.files(input_dir, pattern = "\\.rwl$", full.names = TRUE)

n <- length(rwl_files)  # Total files
for (i in seq_along(rwl_files)) {
  file <- rwl_files[i]
  file_name <- basename(file)
  
  # Progress percentage
  pct <- round(i / n * 100, 1)
  message(sprintf("⏳ Processing (%d/%d, %.1f%%): %s", i, n, pct, file_name))
  
  # Extract continent and site code
  continent <- str_extract(file_name, "^[^_]+")
  site_code <- str_extract(file_name, "(?<=_)[^.]+")
  
  # Construct output file path
  output_name <- tools::file_path_sans_ext(file_name)
  output_csv <- file.path(output_dir, paste0(output_name, ".csv"))
  
  # Main logic with error handling
  tryCatch({
    file = 'F:/Global_ITRDB_RWI/rwl_data/asia_chin067.rwl'
    rwl <- read.rwl(file, format = "tucson")
    rwi <- detrend(rwl, method = "ModNegExp")
    crn <- chron(rwi)
    
    # Save chronology as CSV
    write.csv(cbind(Year = rownames(crn), crn), output_csv, row.names = FALSE)
    
    # Extract statistical metrics
    stats <- rwi.stats(rwi)
    eps_vec <- stats$eps
    eps_mean <- mean(eps_vec, na.rm = TRUE)
    eps_good <- eps_mean > 0.8
    
    # Get crn year information
    crn_years <- as.numeric(rownames(crn))
    year_start <- min(crn_years, na.rm = TRUE)
    year_end   <- max(crn_years, na.rm = TRUE)
    
    # Filter years from 1982 onwards
    post1982 <- crn_years >= 1982
    std_post1982 <- crn[post1982, "std"]
    depth_post1982 <- crn[post1982, "samp.depth"]
    
    # Check for non-NA std
    has_post1982_data <- any(!is.na(std_post1982))
    depth_post1982_ok <- all(depth_post1982 >= 20, na.rm = TRUE)
    
    # Aggregate all fields
    summary_list[[length(summary_list) + 1]] <- data.frame(
      Continent = continent,
      Site_Code = site_code,
      EPS_Mean = round(eps_mean, 4),
      EPS_OK = eps_good,
      Year_Start = year_start,
      Year_End = year_end,
      Has_Post1982 = has_post1982_data,
      Post1982_Depth_greater_20 = depth_post1982_ok,
      stats
    )
    
    
  }, error = function(e) {
    error_count <<- error_count + 1
    cat(
      paste0("\n\n============== Error #", error_count, " ==============\n"),
      "File path: ", file, "\n",
      "Error content: ", conditionMessage(e), "\n",
      file = error_log_path, append = TRUE
    )
    message("❌ Error, logging to file: ", file_name)
  })
}

# Aggregate and save statistical results
summary_df <- do.call(rbind, summary_list)
write.csv(summary_df, summary_csv, row.names = FALSE)
message("✅ All processing complete, results saved to: ", summary_csv)


# Filtering and requested data -----
# Load necessary packages
library(readr)
library(fs)

# --- Path settings ---
summary_csv <- "F:/Global_ITRDB_RWI/chron_statistics_summary.csv"
csv_dir <- "F:/Global_ITRDB_RWI/chron_csv"
rwl_dir <- "F:/Global_ITRDB_RWI/rwl_data"
txt_dir <- "F:/Global_ITRDB_RWI/txt_data"

csv_out <- "E:/GlobalTreeRing/Data/selected_csv"
rwl_out <- "E:/GlobalTreeRing/Data/selected_rwl"
txt_out <- "E:/GlobalTreeRing/Data/selected_txt"
log_file <- "E:/GlobalTreeRing/Data/missing_files_log.txt"

dir_create(c(csv_out, rwl_out, txt_out))
if (file_exists(log_file)) file_delete(log_file)

# --- Read filter conditions ---
summary_data <- read_csv(summary_csv, show_col_types = FALSE)
filtered <- summary_data[summary_data$EPS_OK == TRUE & summary_data$Has_Post1982 == TRUE, ]

# --- Iterate through selected sites ---
for (i in seq_len(nrow(filtered))) {
  continent <- filtered$Continent[i]
  site_code <- filtered$Site_Code[i]
  base_name <- paste0(continent, "_", site_code)
  
  # Construct file paths
  file_csv <- file.path(csv_dir, paste0(base_name, ".csv"))
  file_rwl <- file.path(rwl_dir, paste0(base_name, ".rwl"))
  file_txt <- file.path(txt_dir, paste0(base_name, "-rwl-noaa.txt"))
  
  # Construct target paths
  out_csv <- file.path(csv_out, paste0(base_name, ".csv"))
  out_rwl <- file.path(rwl_out, paste0(base_name, ".rwl"))
  out_txt <- file.path(txt_out, paste0(base_name, ".txt"))
  
  # Copy or log missing files
  if (file_exists(file_csv)) {
    file_copy(file_csv, out_csv, overwrite = TRUE)
    message("✅ Output:", file_csv)
  } else {
    write(paste0("❌ Missing CSV: ", file_csv, "\n------"), file = log_file, append = TRUE)
  }
  
  if (file_exists(file_rwl)) {
    file_copy(file_rwl, out_rwl, overwrite = TRUE)
    message("✅ Output:", file_rwl)
  } else {
    write(paste0("❌ Missing RWL: ", file_rwl, "\n------"), file = log_file, append = TRUE)
  }
  
  if (file_exists(file_txt)) {
    file_copy(file_txt, out_txt, overwrite = TRUE)
    message("✅ Output:", file_txt)
  } else {
    write(paste0("❌ Missing TXT: ", file_txt, "\n------"), file = log_file, append = TRUE)
  }
}


# Filter site and species info from txt -----
library(stringr)
library(readr)

# Set paths
txt_dir <- "E:/GlobalTreeRing/Data/selected_txt"
output_csv <- "E:/GlobalTreeRing/Data/txt_metadata_summary.csv"

# Get all .txt files
txt_files <- list.files(txt_dir, pattern = "\\.txt$", full.names = TRUE)

# Initialize empty table
metadata_all <- data.frame()

# Iterate through files
for (i in seq_along(txt_files)) {
  file_path <- txt_files[i]
  file_name <- basename(file_path)
  message(sprintf("🔍 Reading file %d: %s", i, file_name))
  
  # Extract continent and site code
  continent <- str_extract(file_name, "^[^_]+")
  site_code <- str_extract(file_name, "(?<=_).*(?=\\.)")
  
  # Read all lines
  lines <- readLines(file_path, warn = FALSE, encoding = "UTF-8")
  
  # Initialize items as NA
  country <- site_name <- west_lon <- east_lon <- north_lat <- south_lat <- elevation <- time_unit <- species_name <- common_name <- tree_code <- doi <- NA
  
  # Iterate lines and extract by keywords
  for (line in lines) {
    line <- str_trim(line)
    if (str_detect(line, "^#\\s*Location:")) country <- str_remove(line, "^#\\s*Location:\\s*")
    if (str_detect(line, "^#\\s*Site_Name:")) site_name <- str_remove(line, "^#\\s*Site_Name:\\s*")
    if (str_detect(line, "^#\\s*Westernmost_Longitude:")) west_lon <- str_remove(line, "^#\\s*Westernmost_Longitude:\\s*")
    if (str_detect(line, "^#\\s*Easternmost_Longitude:")) east_lon <- str_remove(line, "^#\\s*Easternmost_Longitude:\\s*")
    if (str_detect(line, "^#\\s*Northernmost_Latitude:")) north_lat <- str_remove(line, "^#\\s*Northernmost_Latitude:\\s*")
    if (str_detect(line, "^#\\s*Southernmost_Latitude:")) south_lat <- str_remove(line, "^#\\s*Southernmost_Latitude:\\s*")
    if (str_detect(line, "^#\\s*(Elevation|Elevation_m):")) elevation <- str_remove(line, "^#\\s*(Elevation|Elevation_m):\\s*")
    if (str_detect(line, "^#\\s*Time_Unit:")) time_unit <- str_remove(line, "^#\\s*Time_Unit:\\s*")
    if (str_detect(line, "^#\\s*Species_Name:")) species_name <- str_remove(line, "^#\\s*Species_Name:\\s*")
    if (str_detect(line, "^#\\s*Common_Name:")) common_name <- str_remove(line, "^#\\s*Common_Name:\\s*")
    if (str_detect(line, "^#\\s*Tree_Species_Code:")) tree_code <- str_remove(line, "^#\\s*Tree_Species_Code:\\s*")
    if (str_detect(line, "^#\\s*(DOI|Dataset_DOI):")) doi <- str_remove(line, "^#\\s*(DOI|Dataset_DOI):\\s*")
    if (str_detect(line, "^#\\s*(Archive|Data_Type):")) data_type <- str_remove(line, "^#\\s*(Archive|Data_Type):\\s*")
  }
  
  # Merge into one data row
  metadata_row <- data.frame(
    File_Name = file_name,
    Continent = continent,
    Site_Code = site_code,
    Country = country,
    Site_Name = site_name,
    Western_Longitude = west_lon,
    Eastern_Longitude = east_lon,
    Northern_Latitude = north_lat,
    Southern_Latitude = south_lat,
    Elevation = elevation,
    Time_Unit = time_unit,
    Species_Name = species_name,
    Common_Name = common_name,
    Tree_Species_Code = tree_code,
    DOI = doi,
    Data_Type = data_type,
    stringsAsFactors = FALSE
  )
  print(metadata_row)
  
  # Add to master table
  metadata_all <- rbind(metadata_all, metadata_row)
}

# Write to CSV
write_csv(metadata_all, output_csv)
message("✅ Extraction complete, saved to: ", output_csv)