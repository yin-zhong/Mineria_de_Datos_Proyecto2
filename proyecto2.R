pacman::p_load(tidymodels, tidyverse, naivebayes, caret, kknn, kernlab)
library(pROC)
library(Rtsne)
library(chron)
library(modelr)
library(lubridate)
library(MLmetrics)

#elegir file
filename = file.choose()
data_train = read.csv(filename)

filename = file.choose()
data_eval = read.csv(filename)

#discretizar noshow
data_train <- data_train %>%
  mutate(
    noshow = ifelse(noshow >= 4, 1, 0), 
    noshow = factor(noshow),
    departure_time = chron(times=departure_time),
    date = as.Date(date),
    day = format(date, "%d"),
    month = format(date, "%m"),
    year = format(date, "%Y"))

data_train <- select(data_train, id, fligth_number, month, distance, noshow, pax_midlow, pax_high, pax_midhigh, pax_low, pax_freqflyer, group_bookings, out_of_stock, dom_cnx, int_cnx, p2p, departure_time, capacity, revenues_usd, bookings)

data_eval <- data_eval %>%
  mutate(
    date = as.Date(date),
    day = format(date, "%d"),
    month = format(date, "%m"),
    year = format(date, "%Y"),
    departure_time = chron(times=departure_time))

data_eval <- select(data_eval, id, fligth_number, month, distance, pax_midlow, pax_high, pax_midhigh, pax_low, pax_freqflyer, group_bookings, out_of_stock, dom_cnx, int_cnx, p2p, departure_time, capacity, revenues_usd, bookings)

#omitir na
data_train= na.omit(data_train)
data_eval = na.omit(data_eval)

#omitir duplicados
data_train <- data_train[!duplicated(data_train$id),]
data_eval <- data_eval[!duplicated(data_eval$id),]

#tomar muestra
index_uno <- data_train$noshow == 1
data_uno <- data_train[index_uno,]
data_uno <- data_uno[sample(nrow(data_uno), 2500),]
data_cero <- data_train[!index_uno,]
data_cero <- data_cero[sample(nrow(data_cero), 2500),]
data_train_down <- rbind(data_cero, data_uno)

data_sample_train <- data_train_down[sample(1:nrow(data_train_down)),]

# receta
receta <- 
  recipe(noshow ~ ., data = data_sample_train) %>% 
  update_role(id, fligth_number, new_role = "ID") %>% 
  step_dummy(all_nominal_predictors()) %>% 
  step_zv(all_predictors())


# division datos
data_split <- initial_split(data_sample_train, prop = 3/4)
train_data <- training(data_split)
test_data  <- testing(data_split)


####### MODELO 1 #####################
# definimos modelo de arbol con 2 niveles de profundidad y min 6 nodos por hoja
modelo_arbol <-
  decision_tree(tree_depth = 2, min_n = 6) %>% 
  set_engine("rpart") %>% 
  set_mode("classification")

#####JUSTIFICACION DE ELECCION DE PARAMETROS######

#ARBOL 
params <- expand.grid(tree_depth = 1:5, 
                      min_n = 5:10)

res <- map2_dfr(params$tree_depth, 
                params$min_n, 
                function(x,y)  
                  fitea(decision_tree(tree_depth = x, min_n = y) %>% 
                          set_engine("rpart") %>% 
                          set_mode("classification")))
res <- cbind(res, params)


########### MODELO 2 #################
modelo_svm <- svm_poly(degree = 2) %>% 
  set_engine("kernlab") %>% 
  set_mode("classification") %>% 
  translate()


#####JUSTIFICACION DE ELECCION DE PARAMETROS######

#SVM
fitea_polySVM <- function(grado){
  
  mod <- svm_poly(degree = grado) %>% 
    set_engine("kernlab") %>% 
    set_mode("classification") %>% 
    translate()
  
  modelo_fit <- 
    workflow() %>% 
    add_model(mod) %>% 
    add_recipe(receta) %>% 
    fit(data = train_data)
  
  model_pred <- 
    predict(modelo_fit, test_data) %>% 
    bind_cols(test_data)
  
  F1_Score(model_pred$noshow, model_pred$.pred_class, positive = "0")
}

# testeamos polinomios
fitea_polySVM(1)
fitea_polySVM(2)
fitea_polySVM(3)


############## MODELO 3################

modelo_knn <-
  nearest_neighbor(neighbors = 9) %>% 
  set_engine("kknn") %>% 
  set_mode("classification")

############################


# definimos funcion fitea para ajustar el model
fitea <- function(mod){
  
  modelo_fit <- 
    workflow() %>% 
    add_model(mod) %>% 
    add_recipe(receta) %>% 
    fit(data = train_data)
  
  model_pred <- 
    predict(modelo_fit, test_data) %>% 
    bind_cols(test_data)
  
  
  return(as_data_frame(F1_Score(model_pred$noshow, model_pred$.pred_class, positive = "0")))
  
}


fitea_eval <- function(mod){
  
  modelo_fit <- 
    workflow() %>% 
    add_model(mod) %>% 
    add_recipe(receta) %>% 
    fit(data = data_sample_train)
  
  model_pred <- 
    predict(modelo_fit, data_eval) %>% 
    bind_cols(data_eval)
  
}

#fitear cada modelo
fitea(modelo_arbol)
fitea(modelo_svm)
fitea(modelo_knn)


prediccion_test <- fitea_eval(modelo_arbol)
csv_prediccion = prediccion_test[".pred_class"]
colnames(csv_prediccion) <- c("noshow")
#agregar filas que faltan 
for (x in 1:345) {
  csv_prediccion[nrow(csv_prediccion) + 1, ] = factor(1) 
}
write.csv(csv_prediccion, "predicciones_noshow.csv", row.names = FALSE)



