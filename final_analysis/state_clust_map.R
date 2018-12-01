library(ggplot2)
library(dplyr)
library(googleVis)
require(datasets)
library(munsell)
library(cowplot)
require(gridExtra)
library(scales)
library(haven)
library(RColorBrewer)


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
data <- read.csv('C:/Users/fredl/Documents/repos/state-flu/data/analysis/state_clust.csv', sep="\t")


tmp <- data$State
new <- state.name[match(tmp,state.abb)]
data$fullstate <- tolower(new)


choro <- left_join(
  map_data("state"), 
  data %>% 
    mutate(region=fullstate)
)


base <- ggplot(data=choro, mapping=aes(x=long, y=lat)) +
  coord_fixed(1.3) +
  geom_polygon(color="black", fill="gray")


p1 <- ggplot(choro, aes(long, lat, group=group)) +
  coord_fixed(1.4) +
  geom_polygon(aes(group = group, fill = factor(Cluster)), color = "white") +
  # geom_polygon(color = "white", fill = NA) +
  theme_bw() +
  ditch_the_axes +
  scale_fill_brewer(type='qual',
                     palette='Set2', na.value = "gray60") +
  labs(fill = "Cluster")

p1

ggsave('clust_map.png', units="in", width=12, height=8, dpi=300)
