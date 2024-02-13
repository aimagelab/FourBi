import wandb


class WandbLog(object):

    def __init__(self, experiment_name: str, project, entity, tags=(), dir='/tmp', id=wandb.util.generate_id()):
        self._wandb = wandb
        self._initialized = False
        self._project = project
        self._entity = entity
        self._experiment_name = experiment_name
        self._dir = dir
        self._tags = tags
        self._id = id

    def setup(self, config):
        if self._wandb is None:
            return
        self._initialized = True

        # Configuration
        if self._wandb.run is None:
            self._wandb.init(project=self._project, entity=self._entity, name=self._experiment_name, dir=self._dir,
                             config=config, tags=self._tags, id=self._id, resume="allow")

        # Set up the wandb metrics
        self._wandb.define_metric('train/avg_loss', summary='min')
        self._wandb.define_metric('train/avg_psnr', summary='max')
        self._wandb.define_metric('train/data_time', summary='mean')
        self._wandb.define_metric('train/time_per_iter', summary='mean')

        self._wandb.define_metric('valid/avg_loss', summary='min')
        self._wandb.define_metric('valid/avg_psnr', summary='max')
        self._wandb.define_metric('valid/time', summary='mean')
        self._wandb.define_metric('valid/patience', summary='min')

        self._wandb.define_metric('test/avg_loss', summary='min')
        self._wandb.define_metric('test/avg_psnr', summary='max')
        self._wandb.define_metric('test/time', summary='mean')

    def add_watch(self, model):
        self._wandb.watch(model, log="all")

    def on_log(self, logs=None):
        self._wandb.log(logs)
