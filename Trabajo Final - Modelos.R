#Setup
rm(list = ls(all = TRUE)) 
set.seed(13)

#Packages
install.packages("caret")
install.packages("devtools")
install.packages("caretEnsemble")
install.packages("gbm")
install.packages("xgboost")
install.packages("pROC")
install.packages("plyr")
install.packages("Metrics")
install.packages("ROCR")
install.packages("gplots")
install.packages("MLmetrics")
install.packages("precrec")
install.packages("PRROC")
install.packages("DiagrammeR")
install.packages("party")
install.packages("mlr")
install_github('caretEnsemble', 'zachmayer') #Install zach's caretEnsemble package


#Libraries
library(caret)
library(devtools)
library(caretEnsemble)
library(gbm)
library(xgboost)
library(Metrics)
library(pROC)
library(plyr)
library(ROCR)
library(gplots)
library(MLmetrics) ## Métricas
library(precrec)
library(PRROC)
library(DiagrammeR) ## Visaulización Forest Random Trees
library(party)
library(mlr) ## Balancear Datos

#Data
datos <- read.csv("default of credit card clients.csv")
colnames(datos)[25] <- "DEFAULT"
colnames(datos)[7] <- "PAY_1"
datos$ID = NULL

## Limpieza de datos y Transformaciones
default <- data.frame(datos)
default$DEFAULT[default$DEFAULT == 0] <- "No"
default$DEFAULT[default$DEFAULT == 1] <- "Yes"
default$DEFAULT = as.factor(default$DEFAULT)
default$EDUCATION[default$EDUCATION == 0] <- 4
default$EDUCATION[default$EDUCATION == 5] <- 4
default$EDUCATION[default$EDUCATION == 6] <- 4

str(default)

#Chequemaos por si puede haber algun N/A y el nombre de las variables
sum(datos == ""| is.na(datos))
names(default)


## Definimos modelo completo
## model.matrix(DEFAULT ~ LIMIT_BAL + SEX + EDUCATION + MARRIAGE + AGE + PAY_1 + PAY_2 + PAY_3 + PAY_4 +
##                     PAY_5 + PAY_6 + BILL_AMT1 + BILL_AMT2 + BILL_AMT3 + BILL_AMT4 + BILL_AMT5 +
##                     BILL_AMT6 + PAY_AMT1 + PAY_AMT2 + PAY_AMT3 + PAY_AMT4 + PAY_AMT5 + PAY_AMT6, default) ## Matrix Model 1

## Definimos el modelo sin las variables que causan multicolinealidad
## model.matrix(DEFAULT ~ LIMIT_BAL + SEX + EDUCATION + MARRIAGE + AGE + PAY_1 + PAY_2 + PAY_3 + PAY_4 +
##                     PAY_5 + PAY_6 + BILL_AMT1 +BILL_AMT6 + PAY_AMT1 + PAY_AMT2 + PAY_AMT3 + PAY_AMT4 +
##                     PAY_AMT5 + PAY_AMT6, default) ## Matrix Model 2


#Split train/test
set.seed(13)
inTraining <- createDataPartition(default$DEFAULT, p = .7, list = F, times = 1)
train.default <- default[inTraining, ]
test.default <- default[-inTraining, ]

## Asignamos numero de folds de k-veces cross-validation
folds = 10

# Fijamos el parámetro de control, nos aplicará CV
fitControl <- trainControl(method = 'cv', 
                           number = folds,
                           classProbs = TRUE,
                           summaryFunction = twoClassSummary,
                           savePrediction = TRUE)

## Preprocesamiento de datos para regularizar                         
PP <- c('center', 'scale')


# ENTRENAMIENTO y PRUEBA

## MODELO LOGÍSTICO:
set.seed(13)
logistic.M1 <- train(DEFAULT ~ ., data = train.default,
               method = "glm", family = binomial, metric = "ROC",
               trControl = fitControl, preProcess = PP)

logistic.M1$results
logistic.M1$pred

### Classificación de las probabilidades - Modelo Lógistico ### Datos TRAIN ###
class.train <- data.frame(obs = logistic.M1$pred$obs,
                         Yes = c(logistic.M1$pred$Yes))

class.train$No <- 1 - class.train$Yes
class.train$pred <- factor(ifelse(class.train$Yes >= 0.5, "Yes", "No"))

### Probabilidades del Modelo Logístico ### Datos TRAIN ###
ggplot(class.train, aes(x = Yes)) + 
  geom_histogram(binwidth = .05) + 
  facet_wrap(~ obs) + 
  xlab("Probability of Default")

### Métricas para el modelo lógistico ### Datos TRAIN ###
confusionMatrix(data = class.train$pred, reference = class.train$obs, mode = "everything")
postResample(pred = class.train$pred, obs = class.train$obs)
twoClassSummary(class.train, lev = levels(class.train$obs))
prSummary(class.train, lev = levels(class.train$obs))

### Predicción de los datos y primeras métricas - Modelo lógistico
pred.logistic.M1 <- predict(logistic.M1, newdata = test.default, type = "prob")

### Classificación de las probabilidades - Modelo Lógistico ### Datos TEST ###
class.test <- data.frame(obs = test.default$DEFAULT,
                         Yes = c(pred.logistic.M1$Yes))

class.test$No <- 1 - class.test$Yes
class.test$pred <- factor(ifelse(class.test$Yes >= 0.5, "Yes", "No"))

### Probabilidades del Modelo Logístico ### Datos TEST ###
ggplot(class.test, aes(x = No)) + 
  geom_histogram(binwidth = .05) + 
  facet_wrap(~ obs) + 
  xlab("Probability of Default")

### Métricas para el modelo lógistico ### Datos TEST ###
confusionMatrix(data = class.test$pred, reference = class.test$obs, mode = "everything")
postResample(pred = class.test$pred, obs = class.test$obs)
twoClassSummary(class.test, lev = levels(class.test$obs))
prSummary(class.test, lev = levels(class.test$obs))
pr.logistic.test <- pr.curve(class.test$No, class.test$Yes, curve = T, max.compute = T, min.compute = T, rand.compute = T)
plot(pr.logistic.test, max.plot = T, min.plot = T, rand.plot = T, fill.area = T)

## Decision Tree (rpart2):
### Tunin Parameter: maxdepth (profundidad del árbol)
TreeGrid <-  expand.grid(maxdepth = c(1:20)) ## Fijamos la profundidad del arbol a 20 y probamos

### Train Decision Tree Model
set.seed(13)
Tree.M1 <- train(DEFAULT ~ ., data = train.default,
                method = "rpart2", metric = "ROC",
                trControl = fitControl, preProcess = PP,
                tuneGrid = TreeGrid) 


Tree.M1$results
Tree.M1$bestTune ## Profundidad máxima que optimiza es 1
Tree.M1$finalModel$variable.importance

### Plot del Árbol de Decisiones
plot(Tree.M1$finalModel)
text(Tree.M1$finalModel)

### Plot Variables más Decisivas
barplot(Tree.M1$finalModel$variable.importance, col = "steelblue2", main = "Variables de Importancia", sub = "Decision Tree")

### Classificación de las probabilidades - Modelo Decision Tree ### Datos TRAIN ###
set.seed(13)

class.train <- data.frame(obs = Tree.M1$pred$obs,
                          Yes = c(Tree.M1$pred$Yes))

class.train$No <- 1 - class.train$Yes
class.train$pred <- factor(ifelse(class.train$Yes >= 0.5, "Yes", "No"))

### Probabilidades del Modelo Decision Tree ### Datos TRAIN ###
ggplot(class.train, aes(x = Yes)) + 
  geom_histogram(binwidth = .05) + 
  facet_wrap(~ obs) + 
  xlab("Probability of Default")

### Métricas para el modelo Decision Tree ### Datos TRAIN ###
confusionMatrix(data = class.train$pred, reference = class.train$obs, mode = "everything")
postResample(pred = class.train$pred, obs = class.train$obs)
twoClassSummary(class.train, lev = levels(class.train$obs))
prSummary(class.train, lev = levels(class.train$obs)) ## Obtenemos AUC

### Predicción de los datos y primeras métricas - Modelo Decision Tree
pred.Tree.M1 <- predict(Tree.M1, newdata = test.default, type = "prob")

### Classificación de las probabilidades - Modelo Decision Tree ### Datos TEST ###
class.test <- data.frame(obs = test.default$DEFAULT,
                         Yes = c(pred.Tree.M1$Yes))

class.test$No <- 1 - class.test$Yes
class.test$pred <- factor(ifelse(class.test$Yes >= 0.5, "Yes", "No"))

### Probabilidades del Modelo Decision Tree ### Datos TEST ###
ggplot(class.test, aes(x = Yes)) + 
  geom_histogram(binwidth = .05) + 
  facet_wrap(~ obs) + 
  xlab("Probability of Default")

### Métricas para el Modelo Decision Tree ### Datos TEST ###
confusionMatrix(data = class.test$pred, reference = class.test$obs, mode = "everything")
postResample(pred = class.test$pred, obs = class.test$obs)
twoClassSummary(class.test, lev = levels(class.test$obs))
prSummary(class.test, lev = levels(class.test$obs))
pr.Tree.test <- pr.curve(class.test$No, class.test$Yes, curve = T, max.compute = T, min.compute = T, rand.compute = T)
plot(pr.Tree.test, max.plot = T, min.plot = T, rand.plot = T, fill.area = T)


# Random Forest (rf)

### Tuning Parameters:
rfGrid <- expand.grid(mtry = c(2, 5, 10))

### Train Random Forest Model
set.seed(13)
rf.M1 <- train(DEFAULT ~ ., data = train.default,
                 method = "rf", metric = "ROC",
                 trControl = fitControl, preProcess = PP,
                 tuneGrid = rfGrid, verbose = T) 

rf.M1
rf.M1$results
rf.M1$bestTune ## mtry que optimiza es 5

### Plot Random Forest
cforest(DEFAULT ~ ., data = train.default, controls = cforest_control(mtry = 5, mincriterion = 0))
rf.plot <- randomForest(DEFAULT ~ ., data = train.default, importance=TRUE, ntree=500, mtry = 5, do.trace=100)

### Classificación de las probabilidades - Modelo Random Forest ### Datos TRAIN ###
set.seed(13)
class.train <- data.frame(obs = rf.M1$pred$obs,
                          Yes = c(rf.M1$pred$Yes))

class.train$No <- 1 - class.train$Yes
class.train$pred <- factor(ifelse(class.train$Yes >= 0.5, "Yes", "No"))

### Probabilidades del Modelo Random Forest ### Datos TRAIN ###
ggplot(class.train, aes(x = Yes)) + 
  geom_histogram(binwidth = .05) + 
  facet_wrap(~ obs) + 
  xlab("Probability of Default")

### Métricas para el modelo Random Forest ### Datos TRAIN ###
confusionMatrix(data = class.train$pred, reference = class.train$obs, mode = "everything")
postResample(pred = class.train$pred, obs = class.train$obs)
twoClassSummary(class.train, lev = levels(class.train$obs))
prSummary(class.train, lev = levels(class.train$obs)) ## Obtenemos AUC

### Predicción de los datos y primeras métricas - Modelo Random Forest
set.seed(13)
pred.rf.M1 <- predict(rf.M1, newdata = test.default, type = "prob")

### Classificación de las probabilidades - Modelo Random Forest ### Datos TEST ###
class.test <- data.frame(obs = test.default$DEFAULT,
                         Yes = c(pred.rf.M1$Yes))

class.test$No <- 1 - class.test$Yes
class.test$pred <- factor(ifelse(class.test$Yes >= 0.5, "Yes", "No"))

### Probabilidades del Modelo Random Forest ### Datos TEST ###
ggplot(class.test, aes(x = Yes)) + 
  geom_histogram(binwidth = .05) + 
  facet_wrap(~ obs) + 
  xlab("Probability of Default")

### Métricas para el Modelo Random Forest ### Datos TEST ###
confusionMatrix(data = class.test$pred, reference = class.test$obs, mode = "everything")
postResample(pred = class.test$pred, obs = class.test$obs)
twoClassSummary(class.test, lev = levels(class.test$obs))
prSummary(class.test, lev = levels(class.test$obs))
pr.rf.test <- pr.curve(class.test$No, class.test$Yes, curve = T, max.compute = T, min.compute = T, rand.compute = T)
plot(pr.rf.test, max.plot = T, min.plot = T, rand.plot = T, fill.area = T)


# Stochastic Gradient Boosting (gbm)

### Tuning Parameters:
gbmGrid <-  expand.grid(interaction.depth = c(1, 5), 
                        n.trees = c(10, 100, 200), 
                        shrinkage = 0.1,
                        n.minobsinnode = c(1, 10, 20))

### Train gbm
set.seed(13)
gbm.M1 <- train(DEFAULT ~ ., data = train.default,
                 method = "gbm", metric = "ROC",
                 trControl = fitControl, preProcess = PP,
                 tuneGrid = gbmGrid, verbose = T) 
## Selecting tuning parameters
## Fitting n.trees = 100, interaction.depth = 5, shrinkage = 0.1, n.minobsinnode = 20 on full training set

gbm.M1$results
gbm.M1$bestTune
summary(gbm.M1)

### Plot del Stochastic Gradient Boosting
plot(gbm.M1$finalModel$train.error, col = "blue")
plot(gbm.M1$finalModel$oobag.improve, col = "red")

ggplot(gbm.M1$results, aes(x = as.factor(n.minobsinnode), y = n.trees, size = ROC, color = ROC)) + 
  geom_point() + 
  theme_bw() + 
  scale_size_continuous(guide = "none")

### Classificación de las probabilidades - Modelo Stochastic Gradient Boosting ### Datos TRAIN ###
set.seed(13)
class.train <- data.frame(obs = gbm.M1$pred$obs,
                          Yes = c(gbm.M1$pred$Yes))

class.train$No <- 1 - class.train$Yes
class.train$pred <- factor(ifelse(class.train$Yes >= 0.5, "Yes", "No"))

### Probabilidades del Modelo Stochastic Gradient Boosting ### Datos TRAIN ###
ggplot(class.train, aes(x = Yes)) + 
  geom_histogram(binwidth = .05) + 
  facet_wrap(~ obs) + 
  xlab("Probability of Default")

### Métricas para el modelo Stochastic Gradient Boosting ### Datos TRAIN ###
confusionMatrix(data = class.train$pred, reference = class.train$obs, mode = "everything")
postResample(pred = class.train$pred, obs = class.train$obs)
twoClassSummary(class.train, lev = levels(class.train$obs))
prSummary(class.train, lev = levels(class.train$obs)) ## Obtenemos AUC

### Predicción de los datos y primeras métricas - Modelo Stochastic Gradient Boosting
set.seed(13)
pred.gbm.M1 <- predict(gbm.M1, newdata = test.default, type = "prob")

### Classificación de las probabilidades - Modelo Stochastic Gradient Boosting ### Datos TEST ###
class.test <- data.frame(obs = test.default$DEFAULT,
                         Yes = c(pred.gbm.M1$Yes))

class.test$No <- 1 - class.test$Yes
class.test$pred <- factor(ifelse(class.test$Yes >= 0.5, "Yes", "No"))

### Probabilidades del Modelo Stochastic Gradient Boosting ### Datos TEST ###
ggplot(class.test, aes(x = Yes)) + 
  geom_histogram(binwidth = .05) + 
  facet_wrap(~ obs) + 
  xlab("Probability of Default")

### Métricas para el Modelo Stochastic Gradient Boosting ### Datos TEST ###
confusionMatrix(data = class.test$pred, reference = class.test$obs, mode = "everything")
postResample(pred = class.test$pred, obs = class.test$obs)
twoClassSummary(class.test, lev = levels(class.test$obs))
prSummary(class.test, lev = levels(class.test$obs))
pr.gbm.test <- pr.curve(class.test$No, class.test$Yes, curve = T, max.compute = T, min.compute = T, rand.compute = T)
plot(pr.gbm.test, max.plot = T, min.plot = T, rand.plot = T, fill.area = T)


# eXtreme Gradient Boosting (xgbTree)

### Tunin Parameter eXgBoosting:
xgbmGrid <-  expand.grid(nrounds = 100, ## numero de maximo de iteraciones
                         eta = c(0.01, 0.001), ## equivalente al shrikage en boosting
                         max_depth = c(1, 5, 10, 20, 40), ## profundidad de los arboles + profundidad + prob. overfitting
                         colsample_bytree = c(0.3, 0.5, 0.7), ## [0,1] Mas alto + prob. overfitting // más bajo + prob. underfitting
                         gamma = 1,  ## valor minimo de reducción requerido para una partición extra
                         min_child_weight = 1,
                         subsample = c(0.5, 0.75, 1)) ## subset del training set 0.5 indica aleatorio, previene overfitting // 
                                                      ## 1 coge todos los datos
# xgbmGrid

### Train eXgbm
set.seed(13)
xgbm.M1 <- train(DEFAULT ~ ., data = train.default,
                method = "xgbTree", metric = "ROC",
                trControl = fitControl, preProcess = PP,
                tuneGrid = xgbmGrid) 

## Selecting tuning parameters
## Fitting nrounds = 100, max_depth = 10, eta = 0.01, gamma = 1, colsample_bytree = 0.7, min_child_weight = 1
## and subsample = 0.5 of training set

xgbm.M1$
xgbm.M1$results
xgbm.M1$bestTune
summary(xgbm.M1)

### Plot del eXtreme Gradient Boosting
xgb.plot.tree(feature_names = xgbm.M1$finalModel$params, model = xgbm.M1$finalModel)

### Classificación de las probabilidades - Modelo eXtreme Gradient Boosting ### Datos TRAIN ###
set.seed(13)
class.train <- data.frame(obs = xgbm.M1$pred$obs,
                          Yes = c(xgbm.M1$pred$Yes))

class.train$No <- 1 - class.train$Yes
class.train$pred <- factor(ifelse(class.train$Yes >= 0.5, "Yes", "No"))

### Probabilidades del Modelo eXtreme Gradient Boosting ### Datos TRAIN ###
ggplot(class.train, aes(x = Yes)) + 
  geom_histogram(binwidth = .05) + 
  facet_wrap(~ obs) + 
  xlab("Probability of Default")

### Métricas para el modelo eXtreme Gradient Boosting ### Datos TRAIN ###
confusionMatrix(data = class.train$pred, reference = class.train$obs, mode = "everything")
postResample(pred = class.train$pred, obs = class.train$obs)
twoClassSummary(class.train, lev = levels(class.train$obs))
prSummary(class.train, lev = levels(class.train$obs)) ## Obtenemos AUC
## pr.TreeM1.train <- pr.curve(class.train$No, class.train$Yes, curve = T, max.compute = T, min.compute = T, rand.compute = T) ## Tarda mucho
## plot(pr.TreeM1.train, max.plot = T, min.plot = T, rand.plot = T, fill.area = T) ## Tarda mucho el anterior paso


### Predicción de los datos y primeras métricas - Modelo eXtreme Gradient Boosting
set.seed(13)
pred.xgbm.M1 <- predict(xgbm.M1, newdata = test.default, type = "prob")

### Classificación de las probabilidades - Modelo eXtreme Gradient Boosting ### Datos TEST ###
class.test <- data.frame(obs = test.default$DEFAULT,
                         Yes = c(pred.xgbm.M1$Yes))

class.test$No <- 1 - class.test$Yes
class.test$pred <- factor(ifelse(class.test$Yes >= 0.5, "Yes", "No"))

### Probabilidades del Modelo eXtreme Gradient Boosting ### Datos TEST ###
ggplot(class.test, aes(x = Yes)) + 
  geom_histogram(binwidth = .05) + 
  facet_wrap(~ obs) + 
  xlab("Probability of Default")

### Métricas para el Modelo eXtreme Gradient Boosting ### Datos TEST ###
confusionMatrix(data = class.test$pred, reference = class.test$obs, mode = "everything")
postResample(pred = class.test$pred, obs = class.test$obs)
twoClassSummary(class.test, lev = levels(class.test$obs))
prSummary(class.test, lev = levels(class.test$obs))
pr.xgbm.test <- pr.curve(class.test$No, class.test$Yes, curve = T, max.compute = T, min.compute = T, rand.compute = T)
plot(pr.xgbm.test, max.plot = T, min.plot = T, rand.plot = T, fill.area = T)


# Evaluación Conjunta de Modelos

## Entrenamiento
resamps.train <- resamples(list(Logistic = logistic.M1,
                          Tree = Tree.M1,
                          RandomForest = rf.M1,
                          Boosting = gbm.M1,
                          eXBoosting = xgbm.M1))

resamps.train
summary(resamps.train)
trellis.par.set(caretTheme())
dotplot(resamps.train, metric = "a")


### Evaluación de los modelos en conjunto

pred.eval <- extractPrediction(list(logistic.M1,
                               Tree.M1,
                               rf.M1,
                               gbm.M1,
                               xgbm.M1))

plotObsVsPred(pred.eval, title(main = "Accurancy Evaluation Model"))


################################################# MODELOS SIN MULTICOLINEALIDAD ############################################################

# ENTRENAMIENTO y PRUEBA PARA MODELOS SIN MULTICOLINEALIDAD

## MODELO LOGÍSTICO:
set.seed(13)
logistic.M2 <- train(DEFAULT ~ LIMIT_BAL + SEX + EDUCATION + MARRIAGE + AGE + PAY_1 + PAY_2 + PAY_3 + PAY_4 +
                       PAY_5 + PAY_6 + BILL_AMT1 +BILL_AMT6 + PAY_AMT1 + PAY_AMT2 + PAY_AMT3 + PAY_AMT4 +
                       PAY_AMT5 + PAY_AMT6, data = train.default,
                     method = "glm", family = binomial, metric = "ROC",
                     trControl = fitControl, preProcess = PP)

logistic.M2$results
logistic.M2$coefnames

### Classificación de las probabilidades - Modelo Lógistico ### Datos TRAIN ###
class.train <- data.frame(obs = logistic.M2$pred$obs,
                          Yes = c(logistic.M2$pred$Yes))

class.train$No <- 1 - class.train$Yes
class.train$pred <- factor(ifelse(class.train$Yes >= 0.5, "Yes", "No"))

### Métricas para el modelo lógistico ### Datos TRAIN ###
confusionMatrix(data = class.train$pred, reference = class.train$obs, mode = "everything")
postResample(pred = class.train$pred, obs = class.train$obs)
twoClassSummary(class.train, lev = levels(class.train$obs))
prSummary(class.train, lev = levels(class.train$obs))

### Predicción de los datos y primeras métricas - Modelo lógistico
pred.logistic.M2 <- predict(logistic.M2, newdata = test.default, type = "prob")

### Classificación de las probabilidades - Modelo Lógistico ### Datos TEST ###
class.test <- data.frame(obs = test.default$DEFAULT,
                         Yes = c(pred.logistic.M2$Yes))

class.test$No <- 1 - class.test$Yes
class.test$pred <- factor(ifelse(class.test$Yes >= 0.5, "Yes", "No"))

### Métricas para el modelo lógistico ### Datos TEST ###
confusionMatrix(data = class.test$pred, reference = class.test$obs, mode = "everything")
postResample(pred = class.test$pred, obs = class.test$obs)
twoClassSummary(class.test, lev = levels(class.test$obs))
prSummary(class.test, lev = levels(class.test$obs))
pr.logistic.test2 <- pr.curve(class.test$No, class.test$Yes, curve = T, max.compute = T, min.compute = T, rand.compute = T)
plot(pr.logistic.test2, max.plot = T, min.plot = T, rand.plot = T, fill.area = T)

## Decision Tree (rpart2):
### Tunin Parameter: maxdepth (profundidad del árbol)
TreeGrid <-  expand.grid(maxdepth = c(1:20)) ## Fijamos la profundidad del arbol a 20 y probamos

### Train Decision Tree Model
set.seed(13)
Tree.M2 <- train(DEFAULT ~ LIMIT_BAL + SEX + EDUCATION + MARRIAGE + AGE + PAY_1 + PAY_2 + PAY_3 + PAY_4 +
                   PAY_5 + PAY_6 + BILL_AMT1 +BILL_AMT6 + PAY_AMT1 + PAY_AMT2 + PAY_AMT3 + PAY_AMT4 +
                   PAY_AMT5 + PAY_AMT6, data = train.default,
                 method = "rpart2", metric = "ROC",
                 trControl = fitControl, preProcess = PP,
                 tuneGrid = TreeGrid) 


Tree.M2$results
Tree.M2$bestTune ## Profundidad máxima que optimiza es 1
Tree.M2$finalModel$variable.importance

### Plot del Árbol de Decisiones
plot(Tree.M2$finalModel)
text(Tree.M2$finalModel)

### Plot Variables más Decisivas
barplot(Tree.M2$finalModel$variable.importance, col = "steelblue2", main = "Variables de Importancia", sub = "Decision Tree")

### Classificación de las probabilidades - Modelo Decision Tree ### Datos TRAIN ###
set.seed(13)
class.train <- data.frame(obs = Tree.M2$pred$obs,
                          Yes = c(Tree.M2$pred$Yes))

class.train$No <- 1 - class.train$Yes
class.train$pred <- factor(ifelse(class.train$Yes >= 0.5, "Yes", "No"))

### Métricas para el modelo Decision Tree ### Datos TRAIN ###
confusionMatrix(data = class.train$pred, reference = class.train$obs, mode = "everything")
postResample(pred = class.train$pred, obs = class.train$obs)
twoClassSummary(class.train, lev = levels(class.train$obs))
prSummary(class.train, lev = levels(class.train$obs)) ## Obtenemos AUC

### Predicción de los datos y primeras métricas - Modelo Decision Tree
pred.Tree.M2 <- predict(Tree.M2, newdata = test.default, type = "prob")

### Classificación de las probabilidades - Modelo Decision Tree ### Datos TEST ###
class.test <- data.frame(obs = test.default$DEFAULT,
                         Yes = c(pred.Tree.M2$Yes))

class.test$No <- 1 - class.test$Yes
class.test$pred <- factor(ifelse(class.test$Yes >= 0.5, "Yes", "No"))

### Métricas para el Modelo Decision Tree ### Datos TEST ###
confusionMatrix(data = class.test$pred, reference = class.test$obs, mode = "everything")
postResample(pred = class.test$pred, obs = class.test$obs)
twoClassSummary(class.test, lev = levels(class.test$obs))
prSummary(class.test, lev = levels(class.test$obs))
pr.Tree.test2 <- pr.curve(class.test$No, class.test$Yes, curve = T, max.compute = T, min.compute = T, rand.compute = T)
plot(pr.Tree.test2, max.plot = T, min.plot = T, rand.plot = T, fill.area = T)


# Random Forest (rf)

### Tuning Parameters:
rfGrid.M2 <- expand.grid(mtry = c(5, 10))

### Train Random Forest Model
set.seed(13)
rf.M2 <- train(DEFAULT ~ LIMIT_BAL + SEX + EDUCATION + MARRIAGE + AGE + PAY_1 + PAY_2 + PAY_3 + PAY_4 +
                 PAY_5 + PAY_6 + BILL_AMT1 +BILL_AMT6 + PAY_AMT1 + PAY_AMT2 + PAY_AMT3 + PAY_AMT4 +
                 PAY_AMT5 + PAY_AMT6, data = train.default,
               method = "rf", metric = "ROC",
               trControl = fitControl, preProcess = PP,
               tuneGrid = rfGrid.M2) 

rf.M2
rf.M2$results
rf.M2$bestTune ## mtry que optimiza es 5

### Plot Random Forest
plot(rf.M2$finalModel)
text(rf.M2$finalModel)

### Classificación de las probabilidades - Modelo Random Forest ### Datos TRAIN ###
set.seed(13)
class.train <- data.frame(obs = rf.M2$pred$obs,
                          Yes = c(rf.M2$pred$Yes))

class.train$No <- 1 - class.train$Yes
class.train$pred <- factor(ifelse(class.train$Yes >= 0.5, "Yes", "No"))

### Métricas para el modelo Random Forest ### Datos TRAIN ###
confusionMatrix(data = class.train$pred, reference = class.train$obs, mode = "everything")
postResample(pred = class.train$pred, obs = class.train$obs)
twoClassSummary(class.train, lev = levels(class.train$obs))
prSummary(class.train, lev = levels(class.train$obs)) ## Obtenemos AUC

### Predicción de los datos y primeras métricas - Modelo Random Forest
pred.rf.M2 <- predict(rf.M2, newdata = test.default, type = "prob")

### Classificación de las probabilidades - Modelo Random Forest ### Datos TEST ###
class.test <- data.frame(obs = test.default$DEFAULT,
                         Yes = c(pred.rf.M2$Yes))

class.test$No <- 1 - class.test$Yes
class.test$pred <- factor(ifelse(class.test$Yes >= 0.5, "Yes", "No"))

### Métricas para el Modelo Random Forest ### Datos TEST ###
confusionMatrix(data = class.test$pred, reference = class.test$obs, mode = "everything")
postResample(pred = class.test$pred, obs = class.test$obs)
twoClassSummary(class.test, lev = levels(class.test$obs))
prSummary(class.test, lev = levels(class.test$obs))
pr.rf.test2 <- pr.curve(class.test$No, class.test$Yes, curve = T, max.compute = T, min.compute = T, rand.compute = T)
plot(pr.Tree.test2, max.plot = T, min.plot = T, rand.plot = T, fill.area = T)

# Stochastic Gradient Boosting (gbm)

### Tuning Parameters:
gbmGrid.M2 <-  expand.grid(interaction.depth = 5, 
                           n.trees = 100, 
                           shrinkage = 0.1,
                           n.minobsinnode = 20)

### Train gbm
set.seed(13)
gbm.M2 <- train(DEFAULT ~ LIMIT_BAL + SEX + EDUCATION + MARRIAGE + AGE + PAY_1 + PAY_2 + PAY_3 + PAY_4 +
                  PAY_5 + PAY_6 + BILL_AMT1 +BILL_AMT6 + PAY_AMT1 + PAY_AMT2 + PAY_AMT3 + PAY_AMT4 +
                  PAY_AMT5 + PAY_AMT6, data = train.default,
                method = "gbm", metric = "ROC",
                trControl = fitControl, preProcess = PP,
                tuneGrid = gbmGrid.M2, verbose = T) 
## Selecting tuning parameters
## Fitting n.trees = 100, interaction.depth = 5, shrinkage = 0.1, n.minobsinnode = 20 on full training set

gbm.M2$results
summary(gbm.M2)

### Plot del Stochastic Gradient Boosting
plot(gbm.M2$finalModel$train.error, col = "blue")
plot(gbm.M2$finalModel$oobag.improve, col = "red")

ggplot(gbm.M2$results, aes(x = as.factor(n.minobsinnode), y = n.trees, size = ROC, color = ROC)) + 
  geom_point() + 
  theme_bw() + 
  scale_size_continuous(guide = "none")

### Classificación de las probabilidades - Modelo Stochastic Gradient Boosting ### Datos TRAIN ###
set.seed(13)
class.train <- data.frame(obs = gbm.M2$pred$obs,
                          Yes = c(gbm.M2$pred$Yes))

class.train$No <- 1 - class.train$Yes
class.train$pred <- factor(ifelse(class.train$Yes >= 0.5, "Yes", "No"))

### Métricas para el modelo Stochastic Gradient Boosting ### Datos TRAIN ###
confusionMatrix(data = class.train$pred, reference = class.train$obs, mode = "everything")
postResample(pred = class.train$pred, obs = class.train$obs)
twoClassSummary(class.train, lev = levels(class.train$obs))
prSummary(class.train, lev = levels(class.train$obs)) ## Obtenemos AUC

### Predicción de los datos y primeras métricas - Modelo Stochastic Gradient Boosting
set.seed(13)
pred.gbm.M2 <- predict(gbm.M2, newdata = test.default, type = "prob")

### Classificación de las probabilidades - Modelo Stochastic Gradient Boosting ### Datos TEST ###
class.test <- data.frame(obs = test.default$DEFAULT,
                         Yes = c(pred.gbm.M2$Yes))

class.test$No <- 1 - class.test$Yes
class.test$pred <- factor(ifelse(class.test$Yes >= 0.5, "Yes", "No"))

### Métricas para el Modelo Stochastic Gradient Boosting ### Datos TEST ###
confusionMatrix(data = class.test$pred, reference = class.test$obs, mode = "everything")
postResample(pred = class.test$pred, obs = class.test$obs)
twoClassSummary(class.test, lev = levels(class.test$obs))
prSummary(class.test, lev = levels(class.test$obs))

# eXtreme Gradient Boosting (xgbTree)

### Tunin Parameter eXgBoosting:
xgbmGrid.M2 <-  expand.grid(nrounds = 100, ## numero de maximo de iteraciones
                            eta = 0.01, ## equivalente al shrikage en boosting
                            max_depth = 10, ## profundidad de los arboles + profundidad + prob. overfitting
                            colsample_bytree = 0.7, ## [0,1] Mas alto + prob. overfitting // más bajo + prob. underfitting
                            gamma = 1,  ## valor minimo de reducción requerido para una partición extra
                            min_child_weight = 1,
                            subsample = 0.5) ## subset del training set 0.5 indica aleatorio, previene overfitting // 
## 1 coge todos los datos
# xgbmGrid

### Train eXgbm
set.seed(13)
xgbm.M2 <- train(DEFAULT ~ LIMIT_BAL + SEX + EDUCATION + MARRIAGE + AGE + PAY_1 + PAY_2 + PAY_3 + PAY_4 +
                   PAY_5 + PAY_6 + BILL_AMT1 +BILL_AMT6 + PAY_AMT1 + PAY_AMT2 + PAY_AMT3 + PAY_AMT4 +
                   PAY_AMT5 + PAY_AMT6, data = train.default,
                 method = "xgbTree", metric = "ROC",
                 trControl = fitControl, preProcess = PP,
                 tuneGrid = xgbmGrid.M2) 

## Selecting tuning parameters
## Fitting nrounds = 100, max_depth = 10, eta = 0.01, gamma = 1, colsample_bytree = 0.7, min_child_weight = 1
## and subsample = 0.5 of training set

xgbm.M2
xgbm.M2$results
xgbm.M2$bestTune
summary(xgbm.M2)

### Plot del eXtreme Gradient Boosting
xgb.plot.tree(feature_names = xgbm.M2$finalModel$params, model = xgbm.M1$finalModel)

### Classificación de las probabilidades - Modelo eXtreme Gradient Boosting ### Datos TRAIN ###
set.seed(13)
class.train <- data.frame(obs = xgbm.M2$pred$obs,
                          Yes = c(xgbm.M2$pred$Yes))

class.train$No <- 1 - class.train$Yes
class.train$pred <- factor(ifelse(class.train$Yes >= 0.5, "Yes", "No"))

### Métricas para el modelo eXtreme Gradient Boosting ### Datos TRAIN ###
confusionMatrix(data = class.train$pred, reference = class.train$obs, mode = "everything")
postResample(pred = class.train$pred, obs = class.train$obs)
twoClassSummary(class.train, lev = levels(class.train$obs))
prSummary(class.train, lev = levels(class.train$obs)) ## Obtenemos AUC

### Predicción de los datos y primeras métricas - Modelo eXtreme Gradient Boosting
set.seed(13)
pred.xgbm.M2 <- predict(xgbm.M2, newdata = test.default, type = "prob")

### Classificación de las probabilidades - Modelo eXtreme Gradient Boosting ### Datos TEST ###
class.test <- data.frame(obs = test.default$DEFAULT,
                         Yes = c(pred.xgbm.M2$Yes))

class.test$No <- 1 - class.test$Yes
class.test$pred <- factor(ifelse(class.test$Yes >= 0.5, "Yes", "No"))

### Métricas para el Modelo eXtreme Gradient Boosting ### Datos TEST ###
confusionMatrix(data = class.test$pred, reference = class.test$obs, mode = "everything")
postResample(pred = class.test$pred, obs = class.test$obs)
twoClassSummary(class.test, lev = levels(class.test$obs))
prSummary(class.test, lev = levels(class.test$obs))
pr.xgbm.test2 <- pr.curve(class.test$No, class.test$Yes, curve = T, max.compute = T, min.compute = T, rand.compute = T)
plot(pr.xgbm.test2, max.plot = T, min.plot = T, rand.plot = T, fill.area = T)

#################################### Pruebas Balanceando los Datos ###############################################################


train.over <- oversample()