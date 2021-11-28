# instalación e importación de librerías
devtools::install_github("rstudio/tfprobability")

library(tfprobability)
install_tfprobability()

install_tfprobability(version = "nightly")

library(tfprobability)
library(tensorflow)
library(keras)

tf$compat$v2$enable_v2_behavior()


library(dplyr)
library(tidyr)
library(ggplot2)

# seteo semilla y genero datos

tf$random$set_seed(42)

create_sine_data <- function(n){
  set.seed(32)
  x <- seq(0, 1*2*pi, length.out = n)
  y1 <- 3*sin(x)
  y1 <- c(seq(0,0, length.out= 59),
          y1 + mapply(rnorm, n = seq(1,1, length.out=length(y1)), mean = seq(0,0, length.out=length(y1)), sd = 0.15*abs(y1)),
          seq(0,0, length.out=59))
  x <- c(seq(-3,0, length.out= 59),
         seq(0,3*2*pi, length.out = n),
         seq(3*2*pi,3*2*pi+3,length.out=59))
  y2 <- 0.1*x+1
  y <- y1 + y2
  data <- list('x' = x, 'y' =  y)
  data
}

data <- create_sine_data(2048)
x <- data$x
y <- data$y

plot(x, y)


# primer modelo probabilístico

negloglik <- function(y, model) - (model %>% tfd_log_prob(y))
inputs = layer_input(shape = c(1))

# outputs compose input + dense layers
dist <- inputs  %>% 
  layer_dense(units = 20, activation = 'relu') %>% 
  layer_dense(units = 50, activation = 'relu') %>% 
  layer_dense(units = 20, activation = 'relu') %>% 
  layer_dense(units = 2) %>% 
  layer_distribution_lambda(function(x)
    tfd_normal(loc = x[, 1, drop = FALSE],
               scale = 1e-3 + 0.05 * tf$math$softplus(x[, 2, drop = FALSE])
    )
  )

model_nobay <- keras_model(inputs=inputs, outputs= dist)
model_nobay %>% compile(optimizer = "adam", loss = negloglik)

model_nobay$summary()

# mostrar entrenamiento en vivo
#options(keras.view_metrics = FALSE)

# Training takes some time
history <- model_nobay %>% fit(
  x = x,
  y = y,
  batch_size = 32,
  epochs = 5000,
  verbose = 1)

plot(history)


# load weights of the model from url file
url <- "https://raw.githubusercontent.com/tensorchiefs/dl_book/master/data/model_aleatoric_weights.hdf5"
download.file(url, "model_nobay.hdf5")
load_model_weights_hdf5(model_nobay, "model_nobay.hdf5")

x_pred <- seq(-10, 30, length.out=133) 
for (rep in 1:2){ 
  #Predictions for two runs
  print(model_nobay$predict(x_pred)[1:3,1])
  } 


# begin figure
runs <- 200
nobay_cpd <- np.zeros((runs,len(x_pred)))
for i in tqdm(range(0,runs)):
  nobay_cpd[i,:]=np.reshape(model_nobay.predict(x_pred),len(x_pred))


def make_plot_runs(ax, preds, alpha_data=1,ylim=[-7,8]):
  ax.scatter(x,y,color="steelblue", alpha=alpha_data,marker='.') #observerd 
for i in range(0,preds.shape[0]):
  ax.plot(x_pred, preds[i],color="black",linewidth=.5)
ax.set_ylim(ylim)

ax = plt.subplot()
make_plot_runs(ax, nobay_cpd)
plt.ylim([-5,6])
plt.xlabel("x",size=16)
plt.ylabel("y",size=16)
plt.title('Non-Bayes: predictions for serveral runs')
plt.show()

np.quantile(nobay_cpd, 0.025, axis=0)

def make_plot_runs_avg(ax, preds, alpha_data=1,ylim=[-7,8]):
  ax.scatter(x,y,color="steelblue", alpha=alpha_data,marker='.') #observerd      
ax.set_ylim(ylim)
ax.plot(x_pred,np.mean(preds,axis=0),color="black",linewidth=1.5)
ax.plot(x_pred,np.quantile(preds, 0.025, axis=0),color="red",linewidth=1.5,linestyle="--")
ax.plot(x_pred,np.quantile(preds, 0.975, axis=0),color="red",linewidth=1.5,linestyle="--")

ax = plt.subplot()
make_plot_runs_avg(ax, nobay_cpd)
plt.ylim([-10,10])
plt.xlabel("x",size=16)
plt.ylabel("y",size=16)
plt.title('Non-Bayes: CPD')
plt.show()

# finish figure

x <- x[35:2082]
y <- y[35:2082]


n <- length(y) %>% tf$cast(tf$float32)

kl_div <- function(q, p, unused)
  tfd_kl_divergence(q, p) / n


inputs = layer_input(shape = c(1))

dist = inputs %>%
  layer_dense_flipout(units = 20,
                      bias_posterior_fn = tfp$layers$util$default_mean_field_normal_fn(),
                      bias_prior_fn = tfp$layers$util$default_multivariate_normal_fn,
                      kernel_divergence_fn = kl_div,
                      bias_divergence_fn = kl_div,
                      activation="relu") %>%
  layer_dense_flipout(50,    bias_posterior_fn = tfp$layers$util$default_mean_field_normal_fn(),
                                  bias_prior_fn = tfp$layers$util$default_multivariate_normal_fn,
                                  kernel_divergence_fn = kl_div,
                                  bias_divergence_fn = kl_div,
                                  activation="relu") %>%
  layer_dense_flipout(20,    bias_posterior_fn = tfp$layers$util$default_mean_field_normal_fn(),
                                  bias_prior_fn = tfp$layers$util$default_multivariate_normal_fn,
                                  kernel_divergence_fn = kl_div,
                                  bias_divergence_fn = kl_div,
                                  activation="relu") %>%
  layer_dense_flipout(2,     bias_posterior_fn = tfp$layers$util$default_mean_field_normal_fn(),
                                  bias_prior_fn = tfp$layers$util$default_multivariate_normal_fn,
                                  kernel_divergence_fn = kl_div,
                                  bias_divergence_fn = kl_div) %>%
     layer_distribution_lambda(function(x)
     tfd_normal(loc = x[, 1, drop = FALSE],
               scale = 1e-3 + 0.05 * tf$math$softplus(x[, 2, drop = FALSE])
    )
  ) 


model_vi <- keras_model(inputs=inputs, outputs= dist)
model_vi  %>% compile(optimizer=optimizer_adam(learning_rate = 0.0002), loss = negloglik) 

model_vi$summary()

# Training takes some time
history <- model_vi %>% fit(
  x = x,
  y = y,
  batch_size = 512,
  epochs = 20000,
  verbose = 1)

plot(history)

# load weights of the model from url file
url <- "https://raw.githubusercontent.com/tensorchiefs/dl_book/master/data/model_vi.hdf5"
download.file(url, "model_vi.hdf5")
load_model_weights_hdf5(model_vi, "model_vi.hdf5")

##################################################
for rep in range(2):
  print(model_vi.predict(x_pred)[0:3].T) #Samples from the posteriori predictive distribution, different for each run

for rep in range(2):
  print(model_params.predict(x_pred)[0:3].T) #Samples from the parameters for the post predictive, different for each run

vi_cpd =np.zeros((runs,len(x_pred)))
for i in tqdm(range(0,runs)):
  vi_cpd[i,:]=np.reshape(model_vi.predict(x_pred),len(x_pred))


###################################################3

inputs <- layer_input(shape = c(1))

params_mc <- inputs %>%
             layer_dense(units = 200, activation="relu") %>%
             layer_dropout(rate = 0.1, trainable = T) %>%
             layer_dense(units = 500, activation="relu") %>%
             layer_dropout(rate = 0.1, trainable = T) %>%
             layer_dense(units = 500, activation="relu")%>%
             layer_dropout(rate = 0.1, trainable = T) %>%
             layer_dense(units = 500, activation="relu") %>%
             layer_dropout(rate = 0.1, trainable = T) %>%
             layer_dense(units = 200, activation="relu") %>%
             layer_dropout(rate = 0.1, trainable = T) %>%
             layer_dense(units = 2)

dist_mc = params_mc %>% layer_distribution_lambda(function(x)
                            tfd_normal(loc = x[, 1, drop = FALSE],
                                       scale = tf$math$exp(x[, 2, drop = FALSE]))
                            , name='normal_exp')

model_mc <- keras_model(inputs=inputs, outputs= dist_mc)
model_mc  %>% compile(optimizer=optimizer_adam(learning_rate = 0.0002), loss = negloglik) 

model_mc$summary()

history <- model_mc %>% fit(
  x = x,
  y = y,
  batch_size = 512,
  epochs = 1000,
  verbose = 1)

plot(history)


# load weights of the model from url file
url <- "https://raw.githubusercontent.com/tensorchiefs/dl_book/master/data/model_mc.hdf5", "model_mc.hdf5"
download.file(url, "model_mc.hdf5")
load_model_weights_hdf5(model_mc, "model_vi.hdf5")

###################

mc_cpd =np.zeros((runs,len(x_pred)))
for i in tqdm(range(0,runs)):
  mc_cpd[i,:]=np.reshape(model_mc.predict(x_pred),len(x_pred))


#plt.figure(figsize=(5,25))
f,ax = plt.subplots(2,3,sharex=True, sharey=False,figsize=(15,8))

lines = 5

make_plot_runs(ax[0,0], nobay_cpd[0:lines])
#make_no_bayes_plot(ax[0,1], model_nobay_mean, model_nobay_sd, add_std=False,ylim=[-7,8])
ax[0,0].set_title('Non Bayes')
make_plot_runs_avg(ax[1,0],nobay_cpd)
ax[1,0].legend(('mean','2.5% prec.','97.25% prec.'), loc='lower right')

ax[0,1].set_title('Bayes VI')
make_plot_runs(ax[0,1], vi_cpd[0:lines], ylim=[-7,8])
make_plot_runs_avg(ax[1,1], vi_cpd, ylim=[-7,8])
ax[1,1].legend(('mean','2.5% prec.','97.25% prec.'), loc='lower right')

ax[0,2].set_title('Bayes MC')
make_plot_runs(ax[0,2], mc_cpd[0:lines], ylim=[-7,8])
make_plot_runs_avg(ax[1,2], mc_cpd, ylim=[-7,8])
ax[1,2].legend(('mean','2.5% prec.','97.25% prec.'), loc='lower right')

#ax[0,0].axis('off')
ax[0,0].text(-10,6.5, "Runs",fontsize=15,horizontalalignment='left')

#ax[1,0].axis('off')
ax[1,0].text(-10,6.5, "Summary Stats",fontsize=15,horizontalalignment='left')


plt.savefig('ch08_good_cpd.pdf')

plt.show()