
setwd('E:/GlobalTreeRing')
site_dir <- './Data/Annual_Summary'

n_ahead = 12

if(site_dir == './Data/Annual_Summary'){
  Date_names <- c("Year")
  RUN_name <- paste0('Annual-VARresult')
  VI_names <- "TRW"
}


# Load packages -----
library(vars) # VAR
library(tseries) # adf.test
library(urca) # ur.df() & vecm
library(dplyr)
library(tidyr)
library(reshape2)
library(ggplot2)
library(missForest)
library(zoo)
library(purrr)

# Define functions -----
# 1. ADFtest [Stabilization: test<1pct|5pct|10pct]
ADF_test <- function(data){
  for(typename in c("none","trend","drift")){
    output <- c()
    output_ADF <- c()
    for(i in 1:ncol(data)){
      urt.i <- ur.df(data[,i],type=typename,selectlags='AIC')
      pct_min <- max(summary(urt.i)@cval[1,])
      DF_test <- summary(urt.i)@teststat[1]
      if(DF_test<pct_min){
        output <- c(output,1)
      }else{
        output <- c(output,0)
      }
    }
    print(paste0("-----",typename,"-----"))
    print(output)
    table_output = length(table(output))
    if(table_output == 1){
      output_ADF = c(output_ADF, typename)
    }else{
      output_ADF = "Non-stationary"
    }
  }
  
  print(colnames(data))
  
  if(output_ADF %in% c("trend","drift")){
    output_ADF="both"
  }
  
  if(output_ADF == "drift"){
    output_ADF="const"
  }
  
  print(output_ADF)
  return(output_ADF)
}

# Create directories -----
dir.create('./Out-Table-and-Figure', showWarnings = T)
Datedir_main <- file.path('./Out-Table-and-Figure', Sys.Date())
dir.create(Datedir_main, showWarnings = T)
Datedir <- file.path(Datedir_main, RUN_name)
dir.create(Datedir, showWarnings = FALSE)

# -------------------------------------------
############################################-
#                VAR model                 #
############################################-
# -------------------------------------------

site_paths <- list.files(site_dir, '.csv$', full.name = T)
df.site_info <- read.csv('./Data/site_info_merge_GS.csv')

CLI_names <- c("CO2", "TMP", "PRE", "SR", "VPD")

# Get start time before program begins
start_time <- Sys.time()

for (VI_i in seq_along(VI_names)) {
  # Create index result folder
  VI_name <- VI_names[VI_i]
  VI_dir <- file.path(Datedir, VI_name)
  dir.create(VI_dir, showWarnings = FALSE)
  
  # Create plot folder
  plot_dir <- file.path(VI_dir, "IRF_plot")
  dir.create(plot_dir, showWarnings = FALSE)
  
  # Results of VAR model for each site
  VAR_all_result_list <- list()
  VARmodel_list <- list()
  result_list <- list()
  
  # Summary tables of results
  # (1) Data for running VAR model
  VAR_data_list <- list()
  # (2) ADF unit root test
  ADF_all <- data.frame() 
  # (3) Johansen cointegration test
  johansen_all_list <- list() 
  # (4) Lag order determination
  lag_RS_all <- data.frame()
  # (5) VAR model fitting results
  var_coeff.all <- data.frame()
  var_R2.all <- data.frame()
  # (6) VAR model stability
  root_all <- data.frame()
  # (7) IRF impulse response coefficients
  irf_all <- data.frame()
  # (8) IRF impulse response analysis plots
  irf_plot_list <- list()
  # (9) Lag duration determination
  q_all <- data.frame() # Defining the range where lag tends to 0
  duration_all <- data.frame()
  # (10) Variance decomposition results
  fevd_all <- data.frame()
  # (11) Causality test results
  causal_all <- data.frame()
  
  
  for(site_i in seq_along(site_paths)){ # seq_along(site_paths)
    # Read data
    lab <- as.numeric(unlist(strsplit(basename(site_paths[site_i]), "_"))[3])
    site_parts <- unlist(strsplit(basename(site_paths[site_i]), "_"))[1:2]
    site_name <- paste(site_parts[1], site_parts[2], sep = "_")
    
    # (1) ===== Get data for running VAR model =====
    df <- read.csv(site_paths[site_i])
    if(VI_name == "TRW"){
      Year_start <- 1940
      Year_end <- df$Year[nrow(df)]
      Year_list <- c(Year_start : Year_end)
    }
    data_date <- df[which(df$Year %in% Year_list), c(Date_names, VI_name, CLI_names)]
    data <- data_date[, c(VI_name, CLI_names)]
    
    # Convert 0 to NA
    data[which(data[,VI_name] == 0), VI_name] <- NA
    na_num <- sum(is.na(data))
    season_n <- nrow(data)/length(Year_list)
    
    # Check if any column has identical values
    constant_columns <- names(data)[sapply(data, function(col) all(col == col[1]))]
    if (length(constant_columns) > 0) {
      VAR_all_result_list[[site_i]] <- paste(
        "Data contains constant column(s):", 
        paste(constant_columns, collapse = ", "), 
        "| Skipping file:", site_name, 
        "| GS month:", season_n, 
        "| NAs number:", na_num
      )
      names(VAR_all_result_list)[site_i] <- site_name
      print(VAR_all_result_list[[site_i]])
      next()
    }
    
    # Skip if more than 5 NAs
    if(na_num >= 5){
      VAR_all_result_list[[site_i]] <- paste("Data contains more than 5 NAs, skipping file:", site_name, 
                                             "GS month: ", season_n, "NAs number: ", na_num)
      print(VAR_all_result_list[[site_i]])
      next()
    }
    
    # Skip if growing season is less than 3 months
    if(site_dir == './Data/Monthly_Summary'){
      if(season_n < 3){
        VAR_all_result_list[[site_i]] <- paste("The growing season is shorter than 2 months, skipping file:", site_name,  "GS month: ", season_n)
        names(VAR_all_result_list)[site_i] <- site_name
        message(VAR_all_result_list[[site_name]])
        next()
      }
    }
    
    # Use Random Forest to impute missing values
    if(any(is.na(data))){
      data <- missForest(data)$ximp
      message("VI_name: ", VI_name, "  ||  site_i: ", site_i, " || site: ", site_name, " have NA!")
    }else{
      message("VI_name: ", VI_name, "  ||  site_i: ", site_i, " || site: ", site_name, " no NA!")
    }
    
    # All sites - All results - Master table
    result_list[[1]] <- data
    names(result_list)[1] <- "RunVARmodelData"
    # All sites - Individual results - Master table
    VAR_data_list[[site_i]] <- data
    names(VAR_data_list)[site_i] <- site_name
    
    # (2) ===== Unit root test =====
    lag_type <- ADF_test(data)
    if(lag_type == "Non-stationary"){
      lag_type = "both"
    }
    print(paste0("lag_type: ", lag_type))
    
    # Save unit root test results to table
    ADF_var <- data.frame()
    for (variable_i in 1:ncol(data)) {
      for(typename in c("none","trend","drift")){
        urt.i <- ur.df(data[,variable_i], type=typename, selectlags='AIC')
        
        # Create data frame containing test statistics and critical values
        adf_results <- data.frame(
          VI_name,
          lab,
          site_name,
          Variable_name = colnames(data)[variable_i],
          typename,
          Level = c("1%", "5%", "10%"),
          Test_Statistic = round(summary(urt.i)@teststat[1], 3),
          Critical_Value = c(summary(urt.i)@cval[1,])
        )
        
        # Add stationarity check column, checking if test statistic is less than critical values
        adf_results$Stationary <- adf_results$Test_Statistic < adf_results$Critical_Value
        
        # Merge results
        ADF_var <- rbind(ADF_var, adf_results)
      } 
    }
    # All sites - All results - Master table
    result_list[[2]] <- ADF_var
    names(result_list)[2] <- "ADFtestResult"
    # All sites - Individual results - Master table
    ADF_all <- rbind(ADF_all, ADF_var)
    
    # (3) ===== Johansen Cointegration Test =====
    if(all(Date_names == "Year")){
      season_n = NULL
    }
    if(all(c("Year", "Month")  %in% Date_names)){
      season_n = nrow(data)/length(unique(data_date$Year))
    }
    
    var_model <- VAR(data, p = 1, type = lag_type, season = season_n)
    coeff_array <- as.data.frame(var_model$varresult[[4]]$coefficients)
    if(any(is.na(coeff_array))){
      VAR_all_result_list[[i]] <- paste0("VAR model result have NA, skipping file: ", lab_group[i])
      print(VAR_all_result_list[[i]])
      next()
    }
    
    johansen_test <- tryCatch({
      ca.jo(data, type = "trace", ecdet = "trend", K = 2)
    }, error = function(e) {
      # Print error message
      error_msg <- paste("Johansen test error:", e$message)
      print(error_msg)
      
      # Return a list containing error message
      return(list(error = TRUE, message = error_msg))
    })
    
    # Check if an error occurred
    if(inherits(johansen_test, "list") && !is.null(johansen_test$error) && johansen_test$error) {
      # If error, create error info data frame
      johansen_result_df <- data.frame(
        error = TRUE,
        message = johansen_test$message
      )
    } else {
      johansen_test_summary <- summary(johansen_test)
      
      test_stats <- johansen_test_summary@teststat
      critical_vals_5pct <- johansen_test_summary@cval[, "5pct"]  # Get critical value at 5% significance level
      
      # Create data frame
      johansen_result_df <- data.frame(
        VI_name,
        lab,
        site_name,
        typename = "trend",
        r = paste("r <=", (length(critical_vals_5pct)-1):0),  # Representation of cointegration rank
        Test_Statistic = test_stats,
        Critical_Value_5pct = critical_vals_5pct
      )
      
      # Add cointegration check column
      johansen_result_df$Cointegration <- johansen_result_df$Test_Statistic > johansen_result_df$Critical_Value_5pct
    }
    
    # All sites - All results - Master table
    result_list[[3]] <- johansen_result_df
    names(result_list)[3] <- "Johansen_Result"
    # All sites - Individual results - Master table
    johansen_all_list[[site_i]] <- johansen_result_df
    names(johansen_all_list)[site_i] <- site_name
    
    # (4) ===== Determine lag order =====
    lag_num <- VARselect(data, type = lag_type, lag.max = 5, season = season_n)$selection[3]
    lag_num <- 1
    print(paste("lag num = ", lag_num))
    
    # Store
    lag_RS <- data.frame(VARselect(data, type = lag_type, lag.max = 10)$criteria)
    lag_RS$VI_name <- VI_name
    lag_RS$lab <- lab
    lag_RS$site_name <- site_name
    
    # All sites - All results - Master table
    result_list[[4]] <- lag_RS
    names(result_list)[4] <- "VARmodel_lagnum"
    # All sites - Individual results - Master table
    lag_RS_all <- rbind(lag_RS_all, lag_RS)
    
    # (5) ===== Build VAR model =====
    var_model <- VAR(data, p = lag_num, type = lag_type, season = season_n)
    coeff_array <- as.data.frame(var_model$varresult[[4]]$coefficients)
    if(any(is.na(coeff_array))){
      VAR_all_result_list[[i]] <- paste0("VAR model result have NA, skipping file: ", lab_group[i])
      print(VAR_all_result_list[[i]])
      next()
    }
    var_result <- summary(var_model)
    
    # Regression coefficients for VAR indices
    var_coeff <- data.frame(var_result$varresult[[1]]$coefficients)
    var_coeff$lab <- lab
    var_coeff$site_name <- site_name
    
    # Overall p-value of VAR
    var_summary_text <- capture.output(summary(var_model))
    # Extract p-value from text
    var_p_value_line <- grep("p-value:", var_summary_text, value = TRUE)[1]
    var_p_value <- sub(".*p-value: ", " ", var_p_value_line)
    var_p_value <- as.numeric(var_p_value)
    
    # VAR regression R2
    var_R2 <- data.frame(VI_name = VI_name,
                         lab = lab, 
                         site_name = site_name,
                         R2 = var_result$varresult[[1]]$r.squared,
                         AdjR2 = var_result$varresult[[1]]$adj.r.squared,
                         p_value = var_p_value)
    print("******** VAR MODEL RESULT ! ********")
    
    # All sites - All results - Master table
    result_list[[5]] <- var_R2
    names(result_list)[5] <- "VARmodel_R2&pvalue"
    # All sites - Individual results - Master table
    var_coeff.all <- rbind(var_coeff.all, var_coeff)
    var_R2.all <- rbind(var_R2.all, var_R2)
    
    # (6) ===== Model stability test =====
    var_root_df <- data.frame(lab = lab, var_result$roots)
    # Add cointegration check column
    var_root_df$VARstability <- abs(max(var_root_df[2:ncol(var_root_df), 2])) < 1
    var_root_df$lab <- lab
    var_root_df$site_name <- site_name
    print("******** VAR MODEL ROOT ! ********")
    
    # All sites - All results - Master table
    result_list[[6]] <- var_root_df
    names(result_list)[6] <- "VARmodel_stability"
    # All sites - Individual results - Master table
    root_all <- rbind(root_all, var_root_df)
    
    # (7) ===== Impulse Response Analysis =====
    irf_result <- irf(var_model, response = VI_name, 
                      n.ahead = n_ahead)
    
    # IRF estimates
    irf_est <- as.data.frame(irf_result$irf)
    colnames(irf_est) <- names(irf_result$irf)
    irf_est$Time <- as.factor(c(1:(n_ahead+1)))
    irf_est$VI <- as.factor(VI_name)
    irf_est.m <- melt(irf_est, value.name = 'irf')
    
    # IRF upper bound
    irf_Upper <- as.data.frame(irf_result$Upper)
    colnames(irf_Upper) <- names(irf_result$Upper)
    irf_Upper$Time <- as.factor(c(1:(n_ahead+1)))
    irf_Upper.m <- melt(irf_Upper,value.name = 'Upper')
    
    # IRF lower bound
    irf_Lower <- as.data.frame(irf_result$Lower)
    colnames(irf_Lower) <- names(irf_result$Lower)
    irf_Lower$Time <- as.factor(c(1:(n_ahead+1)))
    irf_Lower.m <- melt(irf_Lower,value.name = 'Lower')
    
    # Merge IRF results
    irf_est_low_up <- data.frame(irf_est.m, 
                                 Upper = irf_Upper.m$Upper,
                                 Lower = irf_Lower.m$Lower)
    irf_est_low_up$lab <- lab
    irf_est_low_up$site_name <- site_name
    
    # All sites - All results - Master table
    result_list[[7]] <- irf_est_low_up
    names(result_list)[7] <- "IRF_Result"
    # All sites - Individual results - Master table
    irf_all <- rbind(irf_all, irf_est_low_up)
    
    # (8) ===== Plot Impulse Response Analysis =====
    p <- ggplot(data = irf_est_low_up) + 
      geom_line(aes(x = as.numeric(Time), y = irf)) +
      geom_line(aes(x = as.numeric(Time), y = Upper), 
                col = 'red') + 
      geom_line(aes(x = as.numeric(Time), y = Lower),
                col = 'red') +
      geom_hline(yintercept = 0, linetype = "dashed", size = 0.5) +
      labs(x = 'Lag period', y = VI_name) + 
      theme_bw() +
      # facet_wrap(~ variable, )
      facet_grid(rows = vars(variable), scales = "free_y")
    ggsave(file.path(plot_dir, paste0("NO", site_i, "_",site_name,".png")), 
           plot = p, 
           width = 10, height = 10, units = "cm", dpi = 600)
    
    # All sites - All results - Master table
    result_list[[8]] <- p
    names(result_list)[8] <- "IRF_plot"
    # All sites - Individual results - Master table
    irf_plot_list[[site_i]] <- p
    names(irf_plot_list)[site_i] <- site_name
    
    # (9) ===== Lag response order [duration] =====
    duration_var <- data.frame()
    q_var <- data.frame()
    for (ncol in 1:(ncol(irf_est)-2)) {
      irf <- round(abs(irf_est[,ncol]),5)
      
      for(lagn in 1:length(irf)){
        q = max(irf)*0.1
        
        lag_a <- abs(irf[(lagn+1)] - irf[lagn])
        lag_b <- abs(irf[(lagn+2)] - irf[(lagn+1)])
        
        if(is.na(lag_a <= q) || is.na(lag_b <= q)){
          q = max(irf)*0.1
          time_lag <- which(irf < q)[1]
          # print("2")
          break()
        }
        
        if(lag_a <= q & lag_b <= q){
          time_lag <- lagn
          # print("3")
          break()
        }
      }
      print(q)
      df.q <- data.frame(VI = VI_name,
                         variable = colnames(irf_est)[ncol],
                         lab = lab,
                         site_name = site_name,
                         q = q)
      q_var <- rbind(q_var, df.q)
      
      # Lag order result
      print(paste0('********** ',colnames(irf_est)[ncol], 
                   ' lag duration = ',
                   time_lag, ' **********'))
      
      # Export to table
      if(is.na(time_lag)){
        time_lag_max <- NA
        time_lag_intensity <- NA
        time_lag_Std <- NA
        time_lag_Stderror <- NA
      }else{
        time_lag_max <- which(abs(irf_est[1:time_lag, ncol]) == 
                                max(abs(irf_est[1:time_lag, ncol])))
        time_lag_intensity <- round(irf_est[time_lag_max, ncol], 5)
        time_lag_Std <- round(sd(irf_est[1:time_lag, ncol]), 5)
        time_lag_Stderror <- round(time_lag_Std/sqrt(length(irf_est[1:time_lag, ncol])), 5)
      }
      
      
      time_lag_c <- c(VI_name, colnames(irf_est)[ncol], lab, site_name)
      time_lag_c <- c(time_lag_c, 
                      time_lag, 
                      time_lag_intensity, 
                      time_lag_Std,
                      time_lag_Stderror,
                      time_lag_max)
      
      duration_var <- rbind(duration_var, time_lag_c)
      colnames(duration_var) <- c('VI','variable', "lab", "site_name",
                                  'duration','intensity',
                                  'std','std_eeror','max')
    }
    
    # All sites - All results - Master table
    result_list[[9]] <- duration_var
    names(result_list)[9] <- "IRF_Duration&Intensity"
    # All sites - Individual results - Master table
    duration_all <- rbind(duration_all, duration_var)
    q_all <- rbind(q_all, q_var)
    
    # (10) ===== Variance Decomposition Analysis =====
    fevd_result <- fevd(var_model, n.ahead = n_ahead)
    fevd_VI <- as.data.frame(fevd_result[[1]])
    fevd_VI$Time <- as.factor(c(1:n_ahead))
    fevd_VI$VI <- as.factor(VI_name)
    fevd_VI.m <- melt(fevd_VI, value.name = "fevd")
    fevd_VI.m$lab <- lab
    fevd_VI.m$site_name <- site_name
    
    # All sites - All results - Master table
    result_list[[10]] <- fevd_VI.m
    names(result_list)[10] <- "FEVD_Result"
    # All sites - Individual results - Master table
    fevd_all <- rbind(fevd_all, fevd_VI.m)
    
    
    # (11) ===== Causal Analysis =====
    causal_result <- causality(var_model)
    causal_df <- data.frame(causal_result$Granger$method,
                            causal_result$Granger$p.value,
                            causal_result$Instant$method,
                            causal_result$Instant$p.value)
    causal_df$lab <- lab
    causal_df$site_name <- site_name
    
    # All sites - All results - Master table
    result_list[[11]] <- causal_df
    names(result_list)[11] <- "Causal_Result"
    # All sites - Individual results - Master table
    causal_all <- rbind(causal_all, causal_df)
    
    # (12) ===== Save VAR model =====
    # All sites - All results - Master table
    result_list[[12]] <- var_model
    names(result_list)[12] <- "VARmodel_fit"
    # All sites - Individual results - Master table
    VARmodel_list[[site_i]] <- var_model
    names(VARmodel_list)[site_i] <- site_name
    
    # (13) ===== Site info =====
    # All sites - All results - Master table
    result_list[[13]] <- df.site_info[which(df.site_info$site_name == site_name), ]
    names(result_list)[13] <- "Site_info"
    
    # ===== Merge all results into list =====
    VAR_all_result_list[[site_i]] <- result_list
    names(VAR_all_result_list)[site_i] <- site_name
    message("======= ", VI_name,  "  || ",site_name, "  OK!", " =======")
  }
  
  # ===== Save overall results =====
  save(VAR_all_result_list, 
       file = file.path(VI_dir, paste0("ALL_", VI_name, "_VAR_all_result_list.RData")))
  save(VARmodel_list, 
       file = file.path(VI_dir, paste0("ALL_", VI_name, "_VARmodel_list.RData")))
  
  # ===== Save individual results =====
  # (1) Data for running VAR model
  save(VAR_data_list, 
       file = file.path(VI_dir, paste0("NO1_", VI_name, "_VAR_data_list.RData")))
  # (2) ADF unit root test
  write.csv(ADF_all, 
            file.path(VI_dir, paste0("NO2_", VI_name, "_ADF_all.csv")))
  # (3) Johansen cointegration test
  save(johansen_all_list, 
       file = file.path(VI_dir, paste0("NO3_", VI_name, "_johansen_all_list.RData")))
  # (4) Lag order determination
  write.csv(lag_RS_all, 
            file.path(VI_dir, paste0("NO4_", VI_name, "_lag_RS_all.csv")))
  # (5) VAR model fitting results
  write.csv(var_coeff.all, 
            file.path(VI_dir, paste0("NO5-1_", VI_name, "_VAR_coeff_all.csv")))
  write.csv(var_R2.all, 
            file.path(VI_dir, paste0("NO5-2_", VI_name, "_VAR_R2_all.csv")))
  # (6) VAR model stability
  write.csv(root_all, 
            file.path(VI_dir, paste0("NO6_", VI_name, "_root_all.csv")))
  # (7) IRF impulse response coefficients
  write.csv(irf_all, 
            file.path(VI_dir, paste0("NO7_", VI_name, "_IRF_all.csv")))
  # (8) IRF impulse response analysis plots
  save(irf_plot_list, 
       file = file.path(VI_dir, paste0("NO8_", VI_name, "_IRF_plot_list.RData")))
  # (9) Lag duration determination
  write.csv(q_all, 
            file.path(VI_dir, paste0("NO9-1_", VI_name, "_IRF_q_all.csv")))
  write.csv(duration_all, 
            file.path(VI_dir, paste0("NO9-2_", VI_name, "_IRF_duration_all.csv")))
  # (10) Variance decomposition results
  write.csv(fevd_all, 
            file.path(VI_dir, paste0("NO10_", VI_name, "_FEVD_all.csv")))
  # (11) Causality test results
  write.csv(causal_all, 
            file.path(VI_dir, paste0("NO11_", VI_name, "_causal_all.csv")))
  
  # Label
  message("======= ", VI_name,  "OK!", " =======")
}

# Get end time after program finishes
end_time <- Sys.time()
run_time <- end_time - start_time
print(run_time)
