
setwd('E:/GlobalTreeRing')
RUN_name <- 'Fig2-PhylogeneticTree-Map'

# Load packages -----
library(dplyr)
library(tidyr)
library(reshape2)
library(ggplot2)
library(patchwork)

library(openxlsx)
library(raster)
library(terra)
library(sf)
library(lubridate)

library(rnaturalearth)
library(rnaturalearthdata)
library(countrycode)

# Create directories -----
dir.create('./Out-Table-and-Figure', showWarnings = T)
Datedir_main <- file.path('./Out-Table-and-Figure', Sys.Date())
dir.create(Datedir_main, showWarnings = T)
Datedir <- file.path(Datedir_main, RUN_name)
dir.create(Datedir, showWarnings = FALSE)
# Read data ------
df.site_info <- read.xlsx( "./. Result/Phylogenetic-Tree-DeleteThreeSpecies/Family_Info/site_info_group_mapping.xlsx")

# (1) ===== Reclassify Köppen-Geiger Climate Data =====
# Read original Köppen-Geiger raster data
r_Koppen <- rast('F:/Global_Koppen_CliamteZone/koppen_geiger_tif/1991_2020/koppen_geiger_0p5.tif')

# Define reclassification matrix
# Original value -> New classification code:
# 1 (AR) = Arid
# 2 (WD) = Warm Dry
# 3 (WH) = Warm Humid
# 4 (CD) = Cold Dry
# 5 (CH) = Cold Humid
# 6 (Polar) = Polar

reclass_matrix <- matrix(
  c(
    # Tropical climates (Af, Am, Aw) -> Warm Humid (WH)
    1, 3,
    2, 3,
    3, 3,
    
    # Arid climates (BWh, BWk, BSh, BSk) -> Arid (AR)
    4, 1,
    5, 1,
    6, 1,
    7, 1,
    
    # Warm climates
    8,  2,   # Csa -> Warm Dry (WD)
    9,  3,   # Csb -> Warm Humid (WH)
    10, 3,   # Csc -> Warm Humid (WH)
    11, 3,   # Cwa -> Warm Humid (WH)
    12, 3,   # Cwb -> Warm Humid (WH)
    13, 3,   # Cwc -> Warm Humid (WH)
    14, 3,   # Cfa -> Warm Humid (WH)
    15, 3,   # Cfb -> Warm Humid (WH)
    16, 3,   # Cfc -> Warm Humid (WH)
    
    # Cold climates
    17, 4,   # Dsa -> Cold Dry (CD)
    18, 4,   # Dsb -> Cold Dry (CD)
    19, 4,   # Dsc -> Cold Dry (CD)
    20, 4,   # Dsd -> Cold Dry (CD)
    21, 4,   # Dwa -> Cold Dry (CD)
    22, 4,   # Dwb -> Cold Dry (CD)
    23, 4,   # Dwc -> Cold Dry (CD)
    24, 4,   # Dwd -> Cold Dry (CD)
    25, 4,   # Dfa -> Cold Dry (CD)
    26, 4,   # Dfb -> Cold Dry (CD)
    27, 5,   # Dfc -> Cold Humid (CH)
    28, 5,   # Dfd -> Cold Humid (CH)
    
    # Polar climates
    29, 6,   # ET -> Polar
    30, 6    # EF -> Polar
  ),
  ncol = 2,
  byrow = TRUE
)

# Execute reclassification
r_reclassified <- classify(r_Koppen, reclass_matrix)

# Set classification attributes (optional, for easier identification)
levels(r_reclassified) <- data.frame(
  id = 1:6,
  climate_zone = c("Arid", "Warm Dry", "Warm Humid", 
                   "Cold Dry", "Cold Humid", "Polar")
)

# Define output path
output_path <- "F:/Global_Koppen_CliamteZone/koppen_geiger_tif/1991_2020/koppen_geiger_0p5_reclassified.tif"

# # Save reclassification result
# writeRaster(
#   r_reclassified,
#   filename = output_path,
#   overwrite = TRUE,  # Overwrite existing file
#   datatype = "INT1U" # Use 1-byte integer to save space
# )

# Verify result
print(r_reclassified)
plot(r_reclassified, main = "Reclassified Climate Zones")

cat("Reclassification complete! File saved to:", output_path)

df.Koppen <- as.data.frame(r_reclassified, xy = TRUE, na.rm = TRUE)
colnames(df.Koppen)[3] <- "type"

# (2) ===== Site Map =====
# Get continent polygons
world_shp <- ne_countries()

# Create color vectors
climate_colors <-c("#6a3be4","#e55e5e","#2db928","#5ffbf1", "#2b80ff", "#333")
radiant_colors <- c("#7c90c8", "#549e34","#ebcc5f", "#92C2DD", "#e5914c")
group_name <- c("Recent angiosperm clade",
                "Modern angiosperm clade",
                "Intermediate angiosperm clade",
                "Ancient gymnosperm clade",
                "Most ancient gymnosperm clade")

# Plotting
map_base <- ggplot() +
  geom_tile(data = df.Koppen, aes(x = x, y = y, fill = type)) +  # Use reclassified data
  geom_hline(yintercept = 0, color = "gray50", linetype = "dashed", linewidth = 0.5) +
  geom_sf(data = world_shp, fill = NA, color = "gray50", linewidth = 0.2) +
  geom_point(data = df.site_info, aes(x = Lon, y = Lat, 
                                      color = ancient_group, shape = ancient_group), size = 1) +
  coord_sf(ylim = c(-60, 90), expand = FALSE) +
  scale_fill_manual(values = alpha(climate_colors, 0.15), na.value = "NA") +
  scale_color_manual(values = radiant_colors,
                     labels = group_name) +
  scale_shape_manual(values = c(20, 18, 15, 2, 1),
                     labels = group_name) +
  labs(x = "Longitude", y = "Latitude", 
       color = "Species\nGroup", shape = "Species\nGroup",
       title = paste0("(", letters[1], ")")) +
  theme_bw() +
  theme(
    plot.title = element_text(size = 12),
    panel.background = element_blank(),  
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    axis.title.x = element_blank(),
    axis.title.y = element_blank()
  ) 

map_base <- map_base +
  guides(
    fill = guide_legend(title = "Köppen-Geiger\nClimate Classification"),
    color = guide_legend(ncol = 2),
    shape = guide_legend(ncol = 2)
  )

map_base <- map_base + theme(legend.position = "none")
ggsave(file.path(Datedir, "Fig2a-TreeRingSite.png"), 
       plot = map_base,
       width = 20, height = 12, units = "cm", dpi = 300)

map_base <- map_base + theme(legend.position = "bottom")
ggsave(file.path(Datedir, "Fig2a-TreeRingSite-legend.png"), 
       plot = map_base,
       width = 26, height = 15, units = "cm", dpi = 300)

# (3) ===== Site End Year Statistics =====
# Load necessary packages
library(scales)

# Check if data exists
if(exists("df.site_info") && "yr_end" %in% names(df.site_info)) {
  
  # Create density plot
  p <- ggplot(df.site_info, aes(x = yr_end)) +
    geom_density(fill = "#4C72B0", color = "#4C72B0", alpha = 0.4, linewidth = 0.8) +
    labs(
      title = paste0("(", letters[2], ")"),
      x = "Year End",
      y = "Density"
    ) +
    scale_x_continuous(expand = c(0, 0)) +  # Remove x-axis padding
    scale_y_continuous(expand = c(0, 0), limits = c(0, 0.063)) +
    theme_classic() +
    theme(
      axis.line = element_line(color = "black", linewidth = 0.5),
      axis.ticks = element_line(color = "black", linewidth = 0.5),
      axis.text = element_text(color = "black", size = 10),  
      plot.title = element_text(color = "black", size = 12), 
      panel.background = element_blank(),
      plot.margin = margin(5, 5, 5, 5)
    ) +
    labs(x = "Year End", y = "Density")
  
  # Add mean/mode line
  # Calculate mode
  mode_year <- as.numeric(names(which.max(table(df.site_info$yr_end))))
  
  # Calculate density peak
  dens <- density(df.site_info$yr_end)
  density_peak <- dens$x[which.max(dens$y)]
  
  # Mark on the plot
  p <- p + 
    geom_vline(xintercept = mode_year, color = "red", linetype = "dashed") +
    annotate("text", x = mode_year, y = 0, 
             label = paste("Mode:", mode_year), 
             color = "red", vjust = -0.5, hjust = -0.1)
  
  # Display plot
  print(p)
  
  # Save plot (optional)
  ggsave(file.path(Datedir, "Fig2b-YearEndDensity.png"), plot = p, 
         width = 5, height = 5, dpi = 600)
  
} else {
  message("Error: Dataframe df.site_info or column yr_end not found")
}

# (4) ===== Site Cumulative Bar Chart [Climate Zones] =====
# Step 1: Convert two dataframes to sf spatial objects
site_sf <- st_as_sf(df.site_info, coords = c("Lon", "Lat"), crs = 4326)
koppen_sf <- st_as_sf(df.Koppen, coords = c("x", "y"), crs = 4326)

# Step 2: Spatial join - Assign nearest Koppen type to each site
matched_sites <- st_join(site_sf, koppen_sf, join = st_nearest_feature)
df2 <- matched_sites %>% subset(ancient_group == "AncientGroup_E_5")
round(table(matched_sites$type, matched_sites$ancient_group)/2466, 4)*100
round(table(df2$type)/nrow(df2), 4)*100
round(table(df2$ancient_group)/nrow(df2), 4)*100
table(matched_sites$ancient_group)

# Check Polar climate data
df.Polar <- matched_sites %>% subset(type == "Polar")
df.Koppen_test <- as.data.frame(r_Koppen, xy = TRUE, na.rm = TRUE)
colnames(df.Koppen_test)[3] <- "type"
df.Koppen_test$type <- as.character(df.Koppen_test$type)

df.Polar <- data.frame(geometry = as.character(df.Polar$geometry)) 
library(stringr)  # Extract numeric values from longitude/latitude strings
df <-  df.Polar %>%
  # Extract longitude (first number in string) and latitude (second number)
  mutate(
    # Core fix: Regex matches integers/decimals (-?\\d+ for integer, (\\.\\d+)? for optional decimal)
    lon_lat = str_extract_all(geometry, "-?\\d+(\\.\\d+)?"),  # Adapts to integer+decimal coordinates
    # Extract first element as longitude (handle mixed int/dec scenarios)
    lon = as.numeric(sapply(lon_lat, function(x) x[1])),
    # Extract second element as latitude
    lat = as.numeric(sapply(lon_lat, function(x) x[2]))
  ) %>%
  dplyr::select(-lon_lat)  # Remove intermediate temporary column

df_classified <- df %>%
  mutate(
    region = case_when(
      # (1) Central Europe Core: Lon 7-11°E, Lat 45-47°N (adapts to integers like 46)
      between(lon, 7, 11) & between(lat, 45, 47) ~ "Europe_Central",
      # (2) Western North America Core: Lon -153~-116°E, Lat 50-68°N
      between(lon, -153, -116) & between(lat, 50, 68) ~ "NorthAmerica_West",
      # (3) Southwest/Central China Core: Lon 74-101°E, Lat 28-38°N (adapts to integer longitude like 100)
      between(lon, 74, 101) & between(lat, 28, 38) ~ "China_Southwest_Central",
      # (4) Scattered Regions: Sites not in the above 3 cores
      TRUE ~ "Scattered_Regions"
    )
  )

# ---------------------- 3. Classification Statistics: Count, Proportion (New NA check) ----------------------
# Basic statistics (including NA value sites to ensure no omissions)
region_stats <- df_classified %>%
  group_by(region) %>%
  summarise(
    site_count = n(),  # Site count per region
    proportion = round(n()/nrow(df_classified)*100, 2),  # Proportion (keep 2 decimal places)
    na_count = sum(is.na(lon) | is.na(lat))  # NA site count per region (theoretically should be 0)
  ) %>%
  arrange(desc(site_count))  # Sort by site count descending

# Rename regions (more intuitive)
region_stats <- region_stats %>%
  mutate(
    region_name = case_when(
      region == "Europe_Central" ~ "Central Europe Core",
      region == "NorthAmerica_West" ~ "Western North America Core",
      region == "China_Southwest_Central" ~ "Southwest/Central China Core",
      region == "Scattered_Regions" ~ "Scattered Regions"
    )
  ) %>%
  dplyr::select(region_name, site_count, proportion, na_count)  # Adjust column order


ggplot() +
  # geom_tile(data = df.Koppen_test, aes(x = x, y = y, fill = type)) +
  geom_hline(yintercept = 0, color = "gray50", linetype = "dashed", linewidth = 0.5) +
  geom_sf(data = world_shp, fill = NA, color = "gray50", linewidth = 0.1) +
  geom_sf(data = df.Polar, aes(color = ancient_group, shape = ancient_group), size = 0.2) +
  coord_sf(ylim = c(-90, 90), expand = FALSE) +
  scale_fill_manual(values = c("29" = "black", "30" = "gray"), na.value = "NA") +
  scale_color_manual(values = radiant_colors,
                     labels = group_name) +
  scale_shape_manual(values = c(20, 18, 15, 2, 1),
                     labels = group_name) +
  labs(x = "Longitude", y = "Latitude",
       color = "Species\nGroup", shape = "Species\nGroup",
       title = paste0("(", letters[1], ")")) +
  theme_bw() +
  theme(
    plot.title = element_text(size = 12),
    panel.background = element_blank(),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    axis.title.x = element_blank(),
    axis.title.y = element_blank()
  )

# Step 3: Count distribution of ancient_group within each Koppen type
table(matched_sites$type, matched_sites$ancient_group)
count_data <- matched_sites %>%
  st_drop_geometry() %>%
  count(type, ancient_group) %>%
  group_by(type) %>%
  mutate(percentage = n / sum(n) * 100) %>%
  ungroup()
write.csv(count_data, file.path(Datedir, "climatezone_stat.csv"))

matched_sites %>% count(ancient_group)

# Step 4: Create cumulative percentage bar chart
radiant_colors <- c("#7c90c8", "#549e34","#ebcc5f", "#92C2DD", "#e5914c")

p_CLI_bar <- ggplot(count_data, aes(x = percentage, y = type, 
                                    fill = ancient_group)) +
  geom_col(position = "stack") +
  scale_x_continuous(expand = expansion(mult = c(0, 0.05))) +
  scale_fill_manual(values = radiant_colors, 
                    label = group_name,
                    name = "Species\nGroup") +
  labs(x = "Percentage of Sites (%)", 
       y = "Climate Classification",
       title = paste0("(", letters[3],")")) +
  theme_bw() +
  theme(
    panel.grid.major.y = element_blank(),
    panel.grid.minor = element_blank(),
    legend.position = "bottom",
    plot.title = element_text(size = 12)
  )

p_CLI_bar <- p_CLI_bar + 
  guides(
    fill = guide_legend(ncol = 3)
  )

ggsave(file.path(Datedir, "Fig2c-ClimateSite.png"), 
       plot = p_CLI_bar, 
       width = 5, height = 5, dpi = 600)


# (5) ===== Site Cumulative Bar Chart [Continents] =====
# Step 1: Get continent info from longitude/latitude
world <- ne_countries(scale = "medium", returnclass = "sf")
site_points <- st_as_sf(df.site_info, coords = c("Lon", "Lat"), crs = 4326)

# Spatial join - Assign country to each site
site_with_country <- st_join(site_points, world, join = st_intersects)

# Use countrycode package to convert country codes to continent names
site_with_country <- site_with_country %>%
  mutate(
    continent = case_when(
      region_un == "Americas" & subregion %in% c("South America") ~ "South America",
      region_un == "Americas" & subregion %in% c("Northern America", "Central America", "Caribbean") ~ "North America",
      TRUE ~ region_un
    )
  )

site_with_country %>% count(continent) %>%
  mutate(percentage = n / sum(n) * 100) %>%
  ungroup()

# Step 2: Count distribution of ancient_group within each continent
continent_data <- site_with_country %>%
  st_drop_geometry() %>%
  count(continent, ancient_group) %>%
  group_by(continent) %>%
  mutate(percentage = n / sum(n) * 100) %>%
  ungroup()

continent_data <- na.omit(continent_data)

write.csv(continent_data, file.path(Datedir, "continent_stat.csv"))


# Add hemisphere column: Determine Northern/Southern Hemisphere based on latitude
site_with_country1 <- df.site_info %>%
  mutate(hemisphere = ifelse(Lat >= 0, "Northern Hemisphere", "Southern Hemisphere"))

# Count sites in each hemisphere
hemisphere_counts <- site_with_country1 %>%
  st_drop_geometry() %>%
  count(hemisphere, ancient_group) %>%
  mutate(percentage = n / sum(n) * 100)

# Step 3: Create continent percentage bar chart
radiant_colors <- c("#7c90c8", "#549e34", "#ebcc5f", "#92C2DD", "#e5914c")

p_continent <- continent_data %>% 
  ggplot(aes(x = percentage, 
             y = reorder(continent, percentage, FUN = sum),  # Sort by total percentage
             fill = ancient_group)) +
  geom_col(position = "stack") +
  scale_x_continuous(expand = expansion(mult = c(0, 0.05))) +
  scale_fill_manual(values = radiant_colors,
                    labels = group_name,
                    name = "Species\nGroup") +
  labs(x = "Percentage of Sites (%)", 
       y = "Continent",
       title = paste0("(",letters[4],")")) +
  theme_bw() +
  theme(
    panel.grid.major.y = element_blank(),
    panel.grid.minor = element_blank(),
    legend.position = "bottom",
    plot.title = element_text(size = 12),
    axis.text.y = element_text(size = 10)
  )

p_continent <- p_continent + 
  guides(
    fill = guide_legend(ncol = 3)
  )

# Save results
ggsave(file.path(Datedir, "Fig2d-ContinentPlot.png"), 
       plot = p_continent, 
       width = 5, 
       height = 5, 
       dpi = 600)


# (6) ===== Combine =====
combined_plot <- p + p_CLI_bar + p_continent +
  plot_layout(guides = "collect") & 
  theme(legend.position = "bottom")
ggsave(file.path(Datedir, "Fig2-TreeRingSite-lower.png"), 
       plot = combined_plot,
       width = 20, height = 8, units = "cm", dpi = 300)
