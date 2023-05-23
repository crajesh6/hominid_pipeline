# ==============================================================================
# Imports
# ==============================================================================
# pip install wandb
import wandb
from wandb.keras import WandbCallback

# ==============================================================================
# W&B logging
# ==============================================================================
# say you have some config -- dictionary of parameters you want to log
wandb_project = config['wandb_project'] # give a project name here
wandb_name = config['wandb_name'] # give a run/trial name here

wandb.init(project=wandb_project, name=wandb_name, config=config)

# ==============================================================================
# Load dataset, build model, something like
# ==============================================================================
model = model_zoo.base_model(**model_config)

model.compile(
    tf.keras.optimizers.Adam(lr=config['lr']),
    loss='mse',
    metrics=[Spearman, pearson_r] # these metrics will be logged using wandb callback
    )

# ==============================================================================
# train model using wandb callback
# ==============================================================================
history = model.fit(
            x_train, y_train,
            epochs=config['num_epochs'],
            batch_size=config['batch_size'],
            shuffle=True,
            validation_data=(x_valid, y_valid),
            callbacks=[WandbCallback(save_model=(False))]
            )

# ==============================================================================
# Evaluate model and log performance on test set
# ==============================================================================
mse, pcc, scc = summary_statistics(model, x_test,  y_test)

wandb.log({
    'MSE': mse,
    'PCC':  pcc,
    'SCC':  scc,
})
