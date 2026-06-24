# Cargar los datos
data = (read.csv(file.choose(), header=T, encoding = "UTF-8"))

# Factores
data$algorithm  <- factor(data$algorithm,  levels = c("blazepose", "movenet"))
data$resolution <- factor(data$resolution, levels = c("high", "medium", "low"))
data$block      <- factor(data$block,      levels = c("indoor", "outdoor"))

# =============================================================================
#  3.1  VARIABLE DE RESPUESTA (GLOBAL)
# =============================================================================

# Estadisticos Descriptivos de PCK
summary(data$pck)

mean(data$pck)
var(data$pck)
sd(data$pck)

min(data$pck)
max(data$pck)

quantile(data$pck, probs = c(0.25,0.5,0.75))

# Histograma PCK
hist(data$pck, breaks = 20, col = "#AED6F1", border = "white",
     xlab = "PCK@0.5 (%)", ylab = "Frecuencia",
     main = "Histograma de PCK@0.5")
abline(v = mean(data$pck),   col = "red",    lwd = 2, lty = 2)

# Bloxplot PCK
boxplot(data$pck, col = "#7FB3D5", ylab = "PCK@0.5 (%)",
        main = "Distribucion global de PCK@0.5")
abline(h = mean(data$pck), col = "red", lty = 2)

# Grafico Orden Temporal
plot(data$orden_corrida, data$pck, pch = 16, cex = 0.4, col = "#5499C7",
     xlab = "Orden de corrida", ylab = "PCK@0.5 (%)",
     main = "PCK@0.5 segun el orden de ejecucion (aleatorizado)")
abline(h = mean(data$pck), col = "red", lty = 2)

# =============================================================================
#  3.2  VARIABLE DE RESPUESTA POR FACTOR Y NIVELES
# =============================================================================

# PCK por factor algo
aggregate(pck ~ algorithm,
          data=data,
          FUN=mean)

aggregate(pck ~ algorithm,
          data=data,
          FUN=sd)

# Boxplot por algo
boxplot(pck ~ algorithm,  data = data, col = c("#F5B041","#52BE80"),
        ylab = "PCK@0.5 (%)", xlab = "Algoritmo",  main = "Por algoritmo")

# PCK por factor res
aggregate(pck ~ resolution,
          data=data,
          FUN=mean)

aggregate(pck ~ resolution,
          data=data,
          FUN=sd)
          
# Boxplot por res
boxplot(pck ~ resolution, data = data, col = c("#EC7063","#F4D03F","#5DADE2"),
        ylab = "PCK@0.5 (%)", xlab = "Resolucion", main = "Por resolucion")

# PCK por bloque
aggregate(pck ~ block,
          data=data,
          FUN=mean)

aggregate(pck ~ block,
          data=data,
          FUN=sd)

# Boxplot por bloque
boxplot(pck ~ block,      data = data, col = c("#AF7AC5","#85C1E9"),
        ylab = "PCK@0.5 (%)", xlab = "Entorno",    main = "Por entorno")

#Boxplot Combinado Algo * Res
boxplot(pck ~ algorithm*resolution,
        data=data,
        las=2)
par(mar = c(8, 4, 4, 1))
boxplot(pck ~ algorithm*resolution, data = data,
        col = rep(c("#F5B041","#52BE80"), times = 3),
        las = 2, ylab = "PCK@0.5 (%)", xlab = "",
        main = "PCK@0.5 por tratamiento (Algoritmo x Resolucion)")
par(mar = c(5, 4, 4, 2))

# 1. Aumentamos el margen inferior de 10 a 14 para dar más espacio al texto largo
par(mar = c(14, 4, 4, 1))

#Boxplot Combinado Algo * Res * Bloc
boxplot(pck ~ resolution + algorithm + block, data = data,
        col = rep(c("#F5B041","#52BE80"), each = 3, times = 2),
        las = 2, 
        cex.axis = 0.75, # <-- NUEVO: Reduce un poco el tamaño de la letra para que quepa bien
        ylab = "PCK@0.5 (%)", xlab = "",
        main = "PCK@0.5 por celda (Algoritmo x Resolucion x Entorno)")

# Restauramos los márgenes por defecto de R
par(mar = c(5, 4, 4, 2))


# =============================================================================
#  3.3 — INTERACCIONES
# =============================================================================

# Algo x Res
interaction.plot(data$resolution, data$algorithm, data$pck,
                 fun = mean, type = "b", pch = c(16, 17), lwd = 2,
                 col = c("#F5B041","#52BE80"),
                 xlab = "Resolucion", ylab = "Media de PCK@0.5 (%)",
                 trace.label = "Algoritmo",
                 main = "Interaccion Algoritmo x Resolucion")

# Algo x Entorno
interaction.plot(data$block, data$algorithm, data$pck,
                 fun = mean, type = "b", pch = c(16, 17), lwd = 2,
                 col = c("#F5B041","#52BE80"),
                 xlab = "Entorno", ylab = "Media de PCK@0.5 (%)",
                 trace.label = "Algoritmo",
                 main = "Interaccion Algoritmo x Entorno")

# Res x Entorno
interaction.plot(data$resolution, data$block, data$pck,
                 fun = mean, type = "b", pch = c(16, 17), lwd = 2,
                 col = c("#AF7AC5","#5DADE2"),
                 xlab = "Resolucion", ylab = "Media de PCK@0.5 (%)",
                 trace.label = "Entorno",
                 main = "Interaccion Resolucion x Entorno")

# revisar que falta del EDA

# ===========================================================================
# MODELO PARAMÉTRICO DE REFERENCIA ANOVA
# ===========================================================================

install.packages("tidyverse")
library(tidyverse)

alpha <- 0.05

mod_anova <- aov(pck ~ algorithm * resolution + block, data = data)

summary(mod_anova)

# ---------------------------------------------------------------------------
# VALIDACIÓN DE SUPUESTOS
# ---------------------------------------------------------------------------

# Histograma de residuales
hist(mod_anova$residuals)

hist(mod_anova$residuals, breaks = 20, col = "#AED6F1", border = "white",
     xlab = "Residuales del Anova", ylab = "Frecuencia",
     main = "Histograma de Residuales del ANOVA")

# QQ Plot de residuales
qqnorm(mod_anova$residuals, col = "#AED6F1", main = "Gráfico QQ de Residuales del ANOVA")
qqline(mod_anova$residuals, col = "red")

# Shapiro-Wilk
shapiro.test(mod_anova$residuals)

#Independencia
plot(mod_anova$residuals, col = "#AED6F1", main = "Grafico de Dispersión de los Residuales")
abline(h = mean(mod_anova$residuals), col = "red", lty = 2)

#Homocedasticidad
plot(mod_anova$fitted, mod_anova$residuals, col = "#AED6F1",
     xlab = "Valores ajustados",
     ylab = "Residuos",
     main = "Gráfico de residuales vs. valores ajustados")
abline(h = 0, col = "red")

#Prueba de levene
install.packages("car")
library(car)


inter <- interaction(data$algorithm, data$resolution)
leveneTest(pck ~ inter, data = data)

# ===========================================================================
# ANÁLISIS NO PARAMÉTRICO: ALIGNED RANK TRANSFORM (ARTool)
# ===========================================================================

install.packages("ARTool")
library(ARTool)

# Se define la transformación con art()
mod_art <- art(pck ~ algorithm * resolution + (1 | block), data = data)

# y luego se ejecuta un anova() sobre la transformación 
mod_art_anova <- anova(mod_art)

print(mod_art_anova, verbose = TRUE)

# eta square
mod_art_anova$eta.sq.part = with(mod_art_anova, (`F` * `Df`) / ((`F` * `Df`) + `Df.res`))

print(mod_art_anova)

# Prueba post-hoc para no paramétrico

#Tuki para la interaccion
tuki <- art.con(mod_art, "algorithm:resolution", adjust = "tukey")
summary(tuki)

#Tuki Algo
tuki_algo <- art.con(mod_art, "algorithm", adjust = "tukey")
summary(tuki_algo)

#Tuki Res
tuki_res <- art.con(mod_art, "resolution", adjust = "tukey")
summary(tuki_res)

# Potencia de la prueba
