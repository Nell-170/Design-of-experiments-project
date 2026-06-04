# Cargar los datos
data = (read.csv(file.choose(), header=T, encoding = "UTF-8"))

# Factores
data$algorithm <- as.factor(data$algorithm)
data$resolution <- as.factor(data$resolution)
data$block <- as.factor(data$block)

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
        col = rep(c("#F5B041","#52BE80"), each = 3),
        las = 2, ylab = "PCK@0.5 (%)", xlab = "",
        main = "PCK@0.5 por tratamiento (Algoritmo x Resolucion)")
par(mar = c(5, 4, 4, 2))

#Boxplot Combinado Algo * Res * Bloc
par(mar = c(10, 4, 4, 1))
boxplot(pck ~ resolution + algorithm + block, data = data,
        col = rep(c("#F5B041","#52BE80"), each = 3, times = 2),
        las = 2, ylab = "PCK@0.5 (%)", xlab = "",
        main = "PCK@0.5 por celda (Algoritmo x Resolucion x Entorno)")
par(mar = c(5, 4, 4, 2))


