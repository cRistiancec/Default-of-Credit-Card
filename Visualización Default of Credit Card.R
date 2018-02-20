### Visualización de datos ###

install.packages("plotly")
install.packages("ggplot2")
install.packages("ggthemes")
install.packages("gdata")
install.packages("dplyr")
install.packages("car")
install.packages("wesanderson")
install.packages("PerformanceAnalytics")
library(PerformanceAnalytics)
library(ggplot2) # Data visualization
library(ggthemes)
library(gdata)
library(dplyr)
library(car)
library(wesanderson) #Paleta de colores Wes Anderson
library(plotly)
library(caret)

## Carga los datos

datos <- read.csv("default of credit card clients.csv")
default <- data.frame(datos)

## Dimensión de la base de datos
dim(default)

## Estructura de los datos
str(default)

## Eliminaré la columna ID, no creo que sea interesante conocer el individuo concreto sino sus características.
default$ID = NULL

## Cambiaré el nombre de la útima columna porque es muy largo y la columna 7 para que tenga un orden lógico de nomenclatura.
colnames(default)[24] <- "DEFAULT"
colnames(default)[6] <- "PAY_1"

## Distribución de los datos por variable
summary(default) ## habrá que dar nuevo formato a las variable qualitativas

## Seguramente los datos sean poco equilibrados, ya que habrá muchos No Default y pocos Default
default$DEFAULT <- as.factor(cut(default$DEFAULT, 2, labels = c("No Default", "Default")))
table(default$DEFAULT)/nrow(default) #Parece que vamos a trabajar con una base de datos no balanceada
barplot(table(default$DEFAULT)/nrow(default), names.arg = c("Not Default", "Default"), col = "steelblue3")

## Factorizamos las columnas cualitativas
default$SEX <- as.factor(cut(default$SEX, 2, labels = c("Hombre", "Mujer")))
default$MARRIAGE <- as.factor(cut(default$MARRIAGE, 4, labels = c("Divorciado", "Casado", "Soltero", "Otros")))

## En EDUCATION hay valores no definidos en la base de datos, creo que lo más correcto es imputarlo a "Otros" categoria 4, ya que son pocos valores
hist(default$EDUCATION, col = "steelblue3")
default$EDUCATION[default$EDUCATION == 0] <- 4
default$EDUCATION[default$EDUCATION == 5] <- 4
default$EDUCATION[default$EDUCATION == 6] <- 4
default$EDUCATION <- as.factor(cut(default$EDUCATION, 4, labels = c("Posgrado", "Universidad", "Secundaria", "Otros")))
barplot(table(default$EDUCATION), col = "steelblue1", title(main = "Histograma Nivel Educativo", xlab = "Nivel Educativo", ylab = "Frequencia"))

## ggplot(boxplot) -> para sexo, en funcion de crédito otorgado y DEFAULT
g1 <- ggplot(data = default, mapping = aes(x = SEX, y = default$LIMIT_BAL, fill = DEFAULT)) + geom_boxplot() + 
  labs(title = "Crédito vs Sexo\n", y = "Crédito Concedido", x = "Sexo") + theme_bw() + 
  theme(plot.title = element_text(size = 20, face = "bold", color = "black", hjust = 0.5)) +
  scale_fill_manual(values = wes_palette(n = 2, name = "Royal1"))

g1
ggplotly(g1)

## ggplot(boxplot) -> para Estado Civil, en funcion de Crédito Concedido y DEFAULT
g2 <- ggplot(data = default, mapping = aes(x = MARRIAGE, y = default$LIMIT_BAL, fill = DEFAULT)) + geom_boxplot() + 
  labs(title = "Crédito vs Estado Civil\n", y = "Crédito Concedido", x = "Estado Civil") + theme_bw() + 
  theme(plot.title = element_text(size = 20, face = "bold", color = "black", hjust = 0.5)) +
  scale_fill_manual(values = wes_palette(n = 4, name = "Royal1"))

g2
ggplotly(g2)

## ggplot(boxplot) -> para Nivel de Educación, en funcion de Crédito Concedido y DEFAULT
g3 <- ggplot(data = default, mapping = aes(x = EDUCATION, y = default$LIMIT_BAL, fill = DEFAULT)) + geom_boxplot() + 
  labs(title = "Crédito vs Nivel Educativo\n", y = "Crédito Concedido", x = "Nivel de Educación") + theme_bw() + 
  theme(plot.title = element_text(size = 20, face = "bold", color = "black", hjust = 0.5)) +
  scale_fill_manual(values = wes_palette(n = 4, name = "Royal1"))

g3
ggplotly(g3)


## Veamos las correlaciones de pearson
corr.default <- cor(subset(datos, select = c(LIMIT_BAL, SEX, EDUCATION, MARRIAGE, AGE, PAY_0, PAY_2, PAY_3, PAY_4,
                                             PAY_5, PAY_6, BILL_AMT1, BILL_AMT2,BILL_AMT3,BILL_AMT4,BILL_AMT5,
                                             BILL_AMT6, PAY_AMT1,PAY_AMT2, PAY_AMT3,PAY_AMT4,PAY_AMT5,PAY_AMT6)))
corrplot(corr.default, method = "number")
##¿Podría haber colinealidad entre los BILL_AMTX? Lo analizaremos en la modelización de los datos y lo tendremos en cuenta.

## visualizamos los datos para ver si puede existir cierta linealidad con las variables que sospechamos
df <- default[ ,c(1, 12:23)]
str(df)
ch.corr <- chart.Correlation(df, histogram = TRUE, pch = 19) ## Por cierto!! tarda mucho en procesar, pero queda un gráfico chulo.

##Puede haber colinealidad en algunas varibales que quizás afecte al modelo, utilizamos función vif (variance infaltion factors)
vif.default <- vif(glm(DEFAULT ~ LIMIT_BAL + SEX + EDUCATION + MARRIAGE + AGE + PAY_1 + PAY_2 + PAY_3 + PAY_4 +
                       PAY_5 + PAY_6 + BILL_AMT1 + BILL_AMT2 + BILL_AMT3 + BILL_AMT4 + BILL_AMT5 +
                       BILL_AMT6 + PAY_AMT1 + PAY_AMT2 + PAY_AMT3 + PAY_AMT4 + PAY_AMT5 + PAY_AMT6, data = default, family = "binomial"))

data.frame(vif.default) ## como sospechabamos hay colinealidad entre las variables BILL_AMTX

vif.default.f <- vif(glm(DEFAULT ~ LIMIT_BAL + SEX + EDUCATION + MARRIAGE + AGE + PAY_1 + PAY_2 + PAY_3 + PAY_4 + PAY_5 + PAY_6 +
                         BILL_AMT1 + BILL_AMT6 + PAY_AMT1 + PAY_AMT2 + PAY_AMT3 + PAY_AMT4 + PAY_AMT5 + PAY_AMT6, 
                         data = default, family = "binomial"))

data.frame(vif.default.f)
