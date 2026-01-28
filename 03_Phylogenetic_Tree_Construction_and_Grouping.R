
setwd('E:/GlobalTreeRing')
RUN_name <- 'Phylogenetic-Tree-DeleteThreeSpecies'

# Load necessary packages -----
library(devtools)
library(picante)
library(ggplot2)
library(ggtree) 
library(treeio)
library(ape) 
# devtools::install_github("jinyizju/V.PhyloMaker")
# devtools::install_github("jinyizju/V.PhyloMaker2")
library(V.PhyloMaker2)

library(dplyr)
library(stringr)
library(openxlsx)
library(readxl)

library(ggnewscale)
library(tidytree)
# devtools::install_github("helixcn/plantlist")
library(plantlist)

# install.packages("adephylo")

# Create directories -----
dir.create('./Out-Table-and-Figure', showWarnings = T)
Datedir_main <- file.path('./Out-Table-and-Figure', Sys.Date())
dir.create(Datedir_main, showWarnings = T)
Datedir <- file.path(Datedir_main, RUN_name)
dir.create(Datedir, showWarnings = FALSE)

# Read data ------
df.site_info_test <- read.csv( "./. Result/site_info_merge_GS.csv")
df.site_info_test <- subset(df.site_info_test, site_name != "asia_tha019") # Error in IRF results
unique(df.site_info_test$Genus)
unique(df.site_info_test$Species_Name)

df.site_info <- df.site_info_test[-which(df.site_info_test$Genus %in% 
                                           c("Liriodendron","Pseudowintera", "Nectandra")), ]
unique(df.site_info$Genus)
unique(df.site_info$Species_Name)

df.site_Genus <- df.site_info[,c("site_name", "Lon", "Lat", "Elevation",
                                 "yr_strat", "yr_end", "Genus", "Species_Name", "Tree_Species_Code")]

# write.xlsx(df.site_info, file.path(Datedir, 'Genus_info.xlsx'))

# Match tree species against the database -----
gen <- as.character(df.site_info$Species_Name)
gen_unique <- unique(gen)
genTPL <- TPL(gen_unique)
genTPL$species_clean <- gsub(" ", "_", genTPL$YOUR_SEARCH)
# gen_list <- taxa.table(genTPL)
# gen_list <- data.frame(gen_list)

colnames(genTPL)[which(colnames(genTPL) == "YOUR_SEARCH")] <- "species"
colnames(genTPL)[which(colnames(genTPL) == "POSSIBLE_GENUS")] <- "genus"
colnames(genTPL)[which(colnames(genTPL) == "FAMILY")] <- "family"
colnames(genTPL)

# write.xlsx(genTPL, file.path(Datedir, 'genTPL_info.xlsx'))

# Construct phylogenetic tree -----
tree <- phylo.maker(sp.list = genTPL, 
                    tree = GBOTB.extended.TPL, 
                    nodes = nodes.info.1.TPL, 
                    scenarios="S3")
phylo_tree <- tree$scenario.3
write.tree(phylo_tree, file.path(Datedir, "phylo_tree.tre"))

# Classify by clustering -----
# Convert tree to binary structure (if not already)
if (!is.binary(phylo_tree)) {
  phylo_tree <- multi2di(phylo_tree)
}

# Use dynamic clustering with cutree (suitable for non-binary structures)
k <- min(5, max(3, floor(sqrt(Ntip(phylo_tree)))))  # Intelligently determine number of groups
hc <- as.hclust(phylo_tree)
groups <- cutree(hc, k = k)

# Create group dataframe
group_df <- data.frame(
  label = names(groups),
  group = as.factor(paste0("CutreeGroup_", letters[groups],"_",groups))
)
table(group_df$group)

# Correctly add group info to the tree
group_info <- split(group_df$label, group_df$group)
grouped_tree <- groupOTU(phylo_tree, group_info)

# Calculate phylogenetic depth for each species -----
# Calculate distance from all nodes to the root
node_depths <- node.depth.edgelength(phylo_tree)

# Create depth dataframe for each species
species_depth <- data.frame(
  species = phylo_tree$tip.label,  # Species name in phylogenetic tree
  tip_node = 1:length(phylo_tree$tip.label),  # Tip node ID
  depth_to_root = NA,  # Distance to root (same for all species)
  phylo_depth = NA,  # Phylogenetic depth (based on divergence time)
  branch_length = NA,  # Terminal branch length
  sister_distance = NA,  # Distance to nearest sister species
  genus = NA,  # Genus
  family = NA,  # Family
  species_clean = NA,  # Cleaned species name
  matched_to_genTPL = FALSE,  # Whether matched to genTPL
  stringsAsFactors = FALSE
)

# Calculate multiple phylogenetic depth metrics for each species
for (i in 1:nrow(species_depth)) {
  current_species <- species_depth$species[i]
  tip_index <- which(phylo_tree$tip.label == current_species)
  
  # 1. Distance to root (Confirm issue)
  species_depth$depth_to_root[i] <- node_depths[tip_index]
  
  # 2. Terminal branch length (Distance from nearest ancestor node to the species)
  # Find parent node of the tip
  parent_node <- phylo_tree$edge[phylo_tree$edge[,2] == tip_index, 1]
  if (length(parent_node) > 0) {
    species_depth$branch_length[i] <- phylo_tree$edge.length[phylo_tree$edge[,2] == tip_index]
    
    # 3. Phylogenetic depth = Distance from root to the nearest ancestor of the species
    species_depth$phylo_depth[i] <- node_depths[parent_node]
  }
  
  # 4. Calculate distance to nearest sister species
  # Get all species except current one
  other_species <- phylo_tree$tip.label[phylo_tree$tip.label != current_species]
  
  if (length(other_species) > 0) {
    # Calculate cophenetic distance to all other species, take minimum
    cophen_matrix <- cophenetic(phylo_tree)
    species_depth$sister_distance[i] <- min(cophen_matrix[current_species, other_species])
  }
  
  # Attempt to match to genTPL data
  # First try direct match with original name
  match_index <- match(species_depth$species[i], genTPL$species_clean)
  
  if (!is.na(match_index)) {
    # Direct match successful
    species_depth$genus[i] <- genTPL$genus[match_index]
    species_depth$family[i] <- genTPL$family[match_index]
    species_depth$species_clean[i] <- genTPL$species_clean[match_index]
    species_depth$matched_to_genTPL[i] <- TRUE
  } else {
    # Try cleaning species name then match
    # Remove underscores, replace with spaces
    cleaned_name <- gsub("_", " ", species_depth$species[i])
    
    # Remove author info, keep only genus and species name
    name_parts <- strsplit(cleaned_name, " ")[[1]]
    if (length(name_parts) >= 2) {
      simple_name <- paste(name_parts[1], name_parts[2])
      species_depth$species_clean[i] <- simple_name
      
      # Try matching cleaned name
      match_index2 <- match(simple_name, genTPL$species_clean)
      
      if (!is.na(match_index2)) {
        species_depth$genus[i] <- genTPL$genus[match_index2]
        species_depth$family[i] <- genTPL$family[match_index2]
        species_depth$matched_to_genTPL[i] <- TRUE
      } else {
        # If still no match, at least extract genus name
        species_depth$genus[i] <- name_parts[1]
        species_depth$species_clean[i] <- simple_name
        species_depth$matched_to_genTPL[i] <- FALSE
      }
    } else {
      species_depth$species_clean[i] <- cleaned_name
      species_depth$matched_to_genTPL[i] <- FALSE
    }
  }
}

# Show match statistics
cat("Match Statistics:\n")
cat("Total species:", nrow(species_depth), "\n")
cat("Successfully matched to genTPL:", sum(species_depth$matched_to_genTPL), "\n")
cat("Match rate:", round(sum(species_depth$matched_to_genTPL)/nrow(species_depth)*100, 1), "%\n\n")

# Show statistics for different depth metrics
cat("Distance to root statistics (should be identical):\n")
cat("Min:", min(species_depth$depth_to_root, na.rm = TRUE), "\n")
cat("Max:", max(species_depth$depth_to_root, na.rm = TRUE), "\n")
cat("Are all values identical:", length(unique(species_depth$depth_to_root)) == 1, "\n\n")

cat("Phylogenetic depth statistics (based on divergence time):\n")
cat("Min depth:", min(species_depth$phylo_depth, na.rm = TRUE), "\n")
cat("Max depth:", max(species_depth$phylo_depth, na.rm = TRUE), "\n")
cat("Mean depth:", round(mean(species_depth$phylo_depth, na.rm = TRUE), 2), "\n")
cat("Median depth:", round(median(species_depth$phylo_depth, na.rm = TRUE), 2), "\n\n")

cat("Terminal branch length statistics:\n")
cat("Min:", min(species_depth$branch_length, na.rm = TRUE), "\n")
cat("Max:", max(species_depth$branch_length, na.rm = TRUE), "\n")
cat("Mean:", round(mean(species_depth$branch_length, na.rm = TRUE), 2), "\n\n")

cat("Distance to nearest sister species statistics:\n")
cat("Min:", min(species_depth$sister_distance, na.rm = TRUE), "\n")
cat("Max:", max(species_depth$sister_distance, na.rm = TRUE), "\n")
cat("Mean:", round(mean(species_depth$sister_distance, na.rm = TRUE), 2), "\n\n")

write.csv(species_depth, file.path(Datedir, "species_depth.csv"))

# Summarize depth distribution by Family (using phylogenetic depth) -----
if (sum(!is.na(species_depth$family)) > 0) {
  family_depth_summary <- species_depth %>%
    filter(!is.na(family)) %>%
    group_by(family) %>%
    summarise(
      count = n(),
      mean_phylo_depth = round(mean(phylo_depth, na.rm = TRUE), 2),
      min_phylo_depth = round(min(phylo_depth, na.rm = TRUE), 2),
      max_phylo_depth = round(max(phylo_depth, na.rm = TRUE), 2),
      mean_sister_dist = round(mean(sister_distance, na.rm = TRUE), 2),
      .groups = 'drop'
    ) %>%
    arrange(desc(count))
  
  cat("Phylogenetic depth statistics by Family (sorted by species count):\n")
  print(head(family_depth_summary, 10))
}

write.csv(family_depth_summary, file.path(Datedir, 'family_depth_summary.csv'))

# Summarize depth distribution by Genus -----
if (sum(!is.na(species_depth$genus)) > 0) {
  genus_depth_summary <- species_depth %>%
    filter(!is.na(genus)) %>%
    group_by(genus) %>%
    summarise(
      count = n(),
      mean_depth = round(mean(phylo_depth, na.rm = TRUE), 2),
      min_depth = round(min(phylo_depth, na.rm = TRUE), 2),
      max_depth = round(max(phylo_depth, na.rm = TRUE), 2),
      .groups = 'drop'
    ) %>%
    arrange(desc(count))
  
  cat("\nPhylogenetic depth statistics by Genus (sorted by species count, top 10):\n")
  print(head(genus_depth_summary, 10))
}

write.csv(genus_depth_summary, file.path(Datedir, 'genus_depth_summary.csv'))

# View unmatched species
unmatched_species <- species_depth[!species_depth$matched_to_genTPL, ]
if (nrow(unmatched_species) > 0) {
  cat("\nSpecies not matched to genTPL (showing top 10):\n")
  print(head(unmatched_species[, c("species", "species_clean", "genus", "phylo_depth")], 10))
}

# Preview final results
cat("\nFinal species_depth dataframe preview:\n")
print(head(species_depth[, c("species", "phylo_depth", "branch_length", "sister_distance", "genus", "family")], 10))

# Recommended depth metrics explanation
cat("\nRecommended depth metrics explanation:\n")
cat("1. phylo_depth: Species divergence time (Recommended for SHAP analysis)\n")
cat("2. sister_distance: Distance to nearest sister species (Reflects evolutionary distinctiveness)\n")
cat("3. branch_length: Terminal branch length (Reflects recent evolutionary changes)\n")

# Calculate phylogenetic depth and reorder -----
# Calculate distance from all nodes to the root
node_depths <- node.depth.edgelength(phylo_tree)

# Modify group_depth structure to include family info
group_depth <- data.frame(
  old_group = levels(group_df$group),
  count = as.numeric(table(group_df$group)),
  mrca_node = NA,
  depth = NA,
  species = NA,
  species_start = NA,
  species_end = NA,
  genus = NA,
  family = NA,          # Store list of all families
  family_start = NA,    # Family of start species
  family_end = NA,      # Family of end species
  dominant_family = NA  # Family with highest percentage in the group
)

# Process each group
for (i in 1:nrow(group_depth)) {
  # Get all species in current group
  group_species <- group_df$label[group_df$group == group_depth$old_group[i]]
  
  # Find MRCA node
  if (length(group_species) == 1) {
    # Single species case
    node_index <- which(phylo_tree$tip.label == group_species)
    group_depth$mrca_node[i] <- node_index
    group_depth$species[i] <- group_species
  } else {
    # Multiple species case
    group_depth$mrca_node[i] <- getMRCA(phylo_tree, group_species)
    group_depth$species_start[i] <- group_species[1]
    group_depth$species_end[i] <- group_species[length(group_species)]
  }
  
  # Get depth of that node
  group_depth$depth[i] <- node_depths[group_depth$mrca_node[i]]
  
  # Match species to genTPL to get family info
  matched_rows <- genTPL[match(group_species, genTPL$species_clean), ]
  
  # Extract family info
  all_families <- unique(matched_rows$family)
  sorted_families <- sort(all_families)
  
  # Extract family of start and end species
  family_start <- matched_rows$family[1]
  family_end <- matched_rows$family[nrow(matched_rows)]
  
  # Determine dominant family (highest percentage)
  family_counts <- table(matched_rows$family)
  dominant_family <- names(family_counts)[which.max(family_counts)]
  dominant_percent <- round(max(family_counts) / length(group_species) * 100, 1)
  
  # Save family info
  group_depth$family[i] <- paste(sorted_families, collapse = ";")
  group_depth$family_start[i] <- family_start
  group_depth$family_end[i] <- family_end
  group_depth$dominant_family[i] <- paste0(dominant_family, " (", dominant_percent, "%)")
}

# Sort by depth ascending (Small depth = Ancient, Large depth = Recent)
group_depth <- group_depth[order(group_depth$depth), ]

# Create new group labels (Group1 = Most Ancient)
group_depth$new_group <- paste0("AncientGroup_", LETTERS[1:nrow(group_depth)],"_", 1:nrow(group_depth))

# Add labels
group_depth <- group_depth %>%
  mutate(age_category = case_when(
    depth < quantile(depth, 0.33) ~ "Ancient",
    depth > quantile(depth, 0.66) ~ "Recent",
    TRUE ~ "Intermediate"
  ))
group_depth

write.xlsx(group_depth, file.path(Datedir, "group_depth_info.xlsx"))

# Update group dataframe (group_df/phylo_tree) -----
group_mapping <- setNames(group_depth$new_group, group_depth$old_group)
group_df$new_group <- group_mapping[as.character(group_df$group)]

# Update group info in phylogenetic tree
phylo_tree$group <- group_df$new_group[match(phylo_tree$tip.label, group_df$label)]

# Set new group as factor (ordered from ancient to recent)
group_df$new_group <- factor(
  group_df$new_group,
  levels = group_depth$new_group
)


# Save family classification results -----
# Create folder for saving family info
family_dir <- file.path(Datedir, "Family_Info")
dir.create(family_dir, showWarnings = FALSE)

# Save family summary info for each group
write.csv(group_depth, file.path(family_dir, "group_family_info.csv"), row.names = FALSE)

# Save full family-group mapping
full_family_info <- data.frame(
  species_clean = group_df$label,
  species = genTPL$species[match(group_df$label, genTPL$species_clean)],
  genus = genTPL$genus[match(group_df$label, genTPL$species_clean)],
  family = genTPL$family[match(group_df$label, genTPL$species_clean)],
  group = group_df$group,
  ancient_group = group_df$new_group
)
write.xlsx(full_family_info, file.path(family_dir, "full_family_group_mapping.xlsx"))

colnames(df.site_info)[15] <- "species"
df.site_info_group <- merge(df.site_info, full_family_info, by = "species")
write.xlsx(df.site_info_group, file.path(family_dir, "site_info_group_mapping.xlsx"))

# Create family frequency table
family_freq <- as.data.frame(table(full_family_info$family))
colnames(family_freq) <- c("Family", "Frequency")
family_freq <- family_freq[order(-family_freq$Frequency), ]
write.csv(family_freq, file.path(family_dir, "family_frequency.csv"), row.names = FALSE)

# Output key information
cat("\n=== Family info saved to the following directory ===\n")
cat(family_dir, "\n")
cat("1. group_family_info.csv - Summary of family info for each group\n")
cat("2. full_family_group_mapping.csv - Family-Group mapping for all species\n")
cat("3. family_frequency.csv - Family frequency statistics\n")


# 1. Visualize group depth ------
ggplot(group_depth, aes(x = new_group, y = depth)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  labs(title = "Phylogenetic Depth of Groups",
       x = "Group (Group1 = Most Ancient)",
       y = "Distance to Root") +
  theme_minimal()

# 2. Create phylogenetic tree with depth information ------
radiant_colors <- c("#7c90c8", "#549e34","#ebcc5f", "#92C2DD", "#e5914c",
                    "#d16ba5") #, "#ADDB88", "#ba83ca"  "#d16ba5"

n_groups <- length(levels(group_df$group))
group_colors <- setNames(radiant_colors[1:n_groups], levels(group_df$new_group))

offset_num = 1.5
barsize_num = 6
extend_num = 0.5
alpha_num = 0.7

ggtree(phylo_tree, layout = "circular", size = 0.5, branch.length = "none") %<+% group_df +
  # Map colors to groups
  aes(color = new_group) +
  geom_tippoint(aes(color = new_group), size = 0.5, show.legend = FALSE) +
  # geom_tiplab(size = 2, offset = 2) +
  scale_color_manual(values = group_colors,
                     name = "Phylogenetic Groups",
                     labels = paste0(group_depth$new_group, " (", round(group_depth$depth, 1), ")")) +
  theme(legend.position = "none") +
  geom_nodepoint(aes(subset = node %in% group_depth$mrca_node),
                 size = 3, shape = 16, color = "#8f8fc3")

p <- ggtree(phylo_tree, layout = "circular", size = 0.5, branch.length = "none") %<+% group_df +
  # Map colors to groups
  aes(color = new_group) +
  geom_tippoint(aes(color = new_group), size = 0.5, show.legend = FALSE) +
  # geom_tiplab(size = 2, offset = 2) +
  scale_color_manual(values = group_colors,
                     name = "Phylogenetic Groups",
                     labels = paste0(group_depth$new_group, " (", round(group_depth$depth, 1), ")")) +
  theme(legend.position = "none") +
  geom_nodepoint(aes(subset = node %in% group_depth$mrca_node),
                 size = 3, shape = 16, color = "#8f8fc3") +
  # Ancient Group 1
  geom_strip(taxa1 = "Lagerstroemia_speciosa_(L.)_Pers.",
             taxa2 = "Cedrela_odorata_L.",
             offset = offset_num,
             barsize = barsize_num,
             extend = extend_num,
             color = alpha(radiant_colors[1], alpha_num)) +
  # Ancient Group 2
  geom_strip(taxa1 = "Alchornea_triplinervia_(Spreng.)_Müll.Arg.",
             taxa2 = "Quercus_robur_L._=_Quercus_pendunculata_Ehrl.",
             offset = offset_num,
             barsize = barsize_num,
             extend = extend_num,
             color = alpha(radiant_colors[2], alpha_num)) +
  # Ancient Group 3
  geom_strip(taxa1 = "Hedera_helix L.",
             taxa2 = "Fraxinus_americana_L.",
             offset = offset_num,
             barsize = barsize_num,
             extend = extend_num,
             color = alpha(radiant_colors[3], alpha_num)) +
  # Ancient Group 4
  geom_strip(taxa1 = "Halocarpus_biformis_(Hook.)_Quinn",
             taxa2 = "Juniperus_occidentalis_Hook.",
             offset = offset_num,
             barsize = barsize_num,
             extend = extend_num,
             color = alpha(radiant_colors[4], alpha_num)) +
  # Ancient Group 5
  geom_strip(taxa1 = "Pinus_koraiensis_Sieb._&_Zucc.",
             taxa2 = "Cedrus_atlantica_(Endl.)_Manetti_ex_Carrière",
             offset = offset_num,
             barsize = barsize_num,
             extend = extend_num,
             color = alpha(radiant_colors[5], alpha_num))
p

# Save plot
ggsave(file.path(Datedir, "phylogenetic_tree_with_depth.png"), 
       p, width = 5, height = 5)
group_depth$family

# Print depth info
cat("\n=== Phylogenetic Group Depths ===\n")
print(group_depth[, c("new_group", "depth")])