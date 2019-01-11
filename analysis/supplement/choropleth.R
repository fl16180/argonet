library(ggplot2)
library(dplyr)
library(googleVis)
require(datasets)
library(munsell)
library(cowplot)
require(gridExtra)
library(scales)
library(extrafont)

# font_import()
wf <- loadfonts(device = "win")



ditch_the_axes <- theme(
  axis.text = element_blank(),
  axis.line = element_blank(),
  axis.ticks = element_blank(),
  panel.border = element_blank(),
  panel.grid = element_blank(),
  axis.title = element_blank()
)

title_theme <- theme(
  plot.title = element_text(size=11, family='DejaVu Sans Mono')
)


# data <- read.csv('D:/fred/repos/state-flu/data/geomap_data.csv')
data <- read.csv('C:/Users/fredl/Documents/repos/state-flu/data/analysis/geomap_data.csv')
data$improvement3[data$improvement3==0] <- 0.4

choro <- left_join(
  map_data("state"), 
  data %>% 
    mutate(region=tolower(State))
)


base <- ggplot(data=choro, mapping=aes(x=long, y=lat)) +
  coord_fixed(1.3) +
  geom_polygon(color="black", fill="gray")

f_root <- function(x) {
  x^(1/100)
}

f_power <- function(x) {
  x^100
}

four_root_trans <- trans_new(name = 'four_root_trans', transform = f_root, inverse = f_power,
          breaks = extended_breaks(), 
          format = format_format(), domain = c(-Inf, Inf))


# tiff('test.tiff', units="in", width=16, height=8, res=300)


# FIRST PLOT
p1 <- ggplot(choro, aes(long, lat, group=group)) +
  coord_fixed(1.4) +
  geom_polygon(aes(group = group, fill = improvement3)) +
  geom_polygon(color = "white", fill = NA) +
  theme_bw() +
  ditch_the_axes +
  scale_fill_gradient(low = mnsl("5R 8/6"), high = mnsl("5R 5/14"), 
                      na.value = rgb(.85,.85,.85), trans="log2",
                      name=NULL) +
  ggtitle("ARGONet improvement (%)") +
  title_theme

# SECOND PLOT
p2 <- ggplot(choro, aes(long, lat, group=group)) +
  coord_fixed(1.4) +
  geom_polygon(aes(group = group, fill = Population)) +
  geom_polygon(color = "white", fill = NA) +
  theme_bw() +
  ditch_the_axes +
  scale_fill_gradient(low = mnsl("5R 8/6"), high = mnsl("5R 5/14"), 
                      na.value = rgb(.85,.85,.85), trans="log2",
                      name=NULL) +
  ggtitle("Population (millions)") +
  title_theme

# THIRD PLOT
p3 <- ggplot(choro, aes(long, lat, group=group)) +
  coord_fixed(1.4) +
  geom_polygon(aes(group = group, fill = GT.terms)) +
  geom_polygon(color = "white", fill = NA) +
  theme_bw() +
  ditch_the_axes +
  scale_fill_gradient(low = mnsl("5R 8/6"), high = mnsl("5R 5/14"), 
                      na.value = rgb(.85,.85,.85), trans="log2",
                      name=NULL) +
  ggtitle("Non-zero GT terms") +
  title_theme

# FOURTH PLOT
p4 <- ggplot(choro, aes(long, lat, group=group)) +
  coord_fixed(1.4) +
  geom_polygon(aes(group = group, fill = athena.coverage)) +
  geom_polygon(color = "white", fill = NA) +
  theme_bw() +
  ditch_the_axes +
  scale_fill_gradient(low = mnsl("5R 8/6"), high = mnsl("5R 5/14"), 
                      na.value = rgb(.85,.85,.85), trans="log10",
                      name=NULL) +
  ggtitle("athenahealth coverage (visits per thousand)") +
  title_theme

# grid.arrange(p1, p2, p3, p4, ncol=2)
plot_grid(p1, p2, p3, p4, labels = c("a","b","c","d"))

ggsave('map.png', units="in", width=16, height=8, dpi=300)
dev.off()
