from datetime import datetime

class TrainLora:
    def __init__(self, bucket_no_upscale, bucket_reso_steps, cache_latents, cache_latents_to_disk, enable_bucket,
                 min_bucket_reso, max_bucket_reso, learning_rate, logging_dir, lr_scheduler, lr_scheduler_num_cycles,
                 max_data_loader_n_workers, max_grad_norm, resolution, max_train_steps, mixed_precision, network_alpha,
                 network_dim, network_module, no_half_vae, optimizer_type, output_dir, output_name,
                 pretrained_model_name_or_path, save_every_n_epochs, save_model_as, save_precision, text_encoder_lr,
                 train_batch_size, train_data_dir, unet_lr, xformers):
        self.start_time = datetime.now()
        self.bucket_no_upscale = bucket_no_upscale
        self.bucket_reso_steps = bucket_reso_steps
        self.cache_latents = cache_latents
        self.cache_latents_to_disk = cache_latents_to_disk
        self.enable_bucket = enable_bucket
        self.min_bucket_reso = min_bucket_reso
        self.max_bucket_reso = max_bucket_reso
        self.learning_rate = learning_rate
        self.logging_dir = logging_dir
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_num_cycles = lr_scheduler_num_cycles
        self.max_data_loader_n_workers = max_data_loader_n_workers
        self.max_grad_norm = max_grad_norm
        self.resolution = resolution
        self.max_train_steps = max_train_steps
        self.mixed_precision = mixed_precision
        self.network_alpha = network_alpha
        self.network_dim = network_dim
        self.network_module = network_module
        self.no_half_vae = no_half_vae
        self.optimizer_type = optimizer_type
        self.output_dir = output_dir
        self.output_name = output_name
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.save_every_n_epochs = save_every_n_epochs
        self.save_model_as = save_model_as
        self.save_precision = save_precision
        self.text_encoder_lr = text_encoder_lr
        self.train_batch_size = train_batch_size
        self.train_data_dir = train_data_dir
        self.unet_lr = unet_lr
        self.xformers = xformers

    def generate_command(self):
        command = [
            'python',
            'sdxl_train_network.py'
        ]
        if self.bucket_no_upscale:
            command.append('--bucket_no_upscale')
        command.append('--bucket_reso_steps={}'.format(self.bucket_reso_steps))
        if self.cache_latents:
            command.append('--cache_latents')
        if self.cache_latents_to_disk:
            command.append('--cache_latents_to_disk')
        if self.enable_bucket:
            command.append('--enable_bucket')
        command.extend([
            '--min_bucket_reso={}'.format(self.min_bucket_reso),
            '--max_bucket_reso={}'.format(self.max_train_steps),
            '--learning_rate={}'.format(self.learning_rate),
            '--logging_dir={}'.format(self.logging_dir),
            '--lr_scheduler={}'.format(self.lr_scheduler),
            '--lr_scheduler_num_cycles={}'.format(self.lr_scheduler_num_cycles),
            '--max_data_loader_n_workers={}'.format(self.max_data_loader_n_workers),
            '--max_grad_norm={}'.format(self.max_grad_norm),
            '--resolution={}'.format(self.resolution),
            '--max_train_steps={}'.format(self.max_train_steps),
            '--mixed_precision={}'.format(self.mixed_precision),
            '--network_alpha={}'.format(self.network_alpha),
            '--network_dim={}'.format(self.network_dim),
            '--network_module={}'.format(self.network_module)
        ])
        if self.no_half_vae:
            command.append('--no_half_vae')

        command.extend([
            '--optimizer_type={}'.format(self.optimizer_type),
            '--output_dir={}'.format(self.output_dir),
            '--output_name={}'.format(self.output_name),
            '--pretrained_model_name_or_path={}'.format(self.pretrained_model_name_or_path),
            '--save_every_n_epochs={}'.format(self.save_every_n_epochs),
            '--save_model_as={}'.format(self.save_model_as),
            '--save_precision={}'.format(self.save_precision),
            '--text_encoder_lr={}'.format(self.text_encoder_lr),
            '--train_batch_size={}'.format(self.train_batch_size),
            '--train_data_dir={}'.format(self.train_data_dir),
            '--unet_lr={}'.format(self.unet_lr)
        ])

        if self.xformers:
            command.append('--xformers')
        print("command: ", command)
        return command

    def get_json(self):
        response = json.dumps({
            'start_time': self.start_time.isoformat(),
            'bucket_no_upscale' : self.bucket_no_upscale,
            'bucket_reso_steps' : self.bucket_reso_steps,
            'cache_latents' : self.cache_latents,
            'cache_latents_to_disk' : self.cache_latents_to_disk,
            'enable_bucket' : self.enable_bucket,
            'min_bucket_reso' : self.min_bucket_reso,
            'max_bucket_reso' : self.max_bucket_reso,
            'learning_rate' : self.learning_rate,
            'logging_dir' : self.logging_dir,
            'lr_scheduler' : self.lr_scheduler,
            'lr_scheduler_num_cycles' : self.lr_scheduler_num_cycles,
            'max_data_loader_n_workers' : self.max_data_loader_n_workers,
            'max_grad_norm' : self.max_grad_norm,
            'resolution' : self.resolution,
            'max_train_steps' : self.max_train_steps,
            'mixed_precision' : self.mixed_precision,
            'network_alpha' : self.network_alpha,
            'network_dim' : self.network_dim,
            'network_module' : self.network_module,
            'no_half_vae' : self.no_half_vae,
            'optimizer_type' : self.optimizer_type,
            'output_dir' : self.output_dir,
            'output_name' : self.output_name,
            'pretrained_model_name_or_path' : self.pretrained_model_name_or_path,
            'save_every_n_epochs' : self.save_every_n_epochs,
            'save_model_as' : self.save_model_as,
            'save_precision' : self.save_precision,
            'text_encoder_lr' : self.text_encoder_lr,
            'train_batch_size' : self.train_batch_size,
            'train_data_dir' : self.train_data_dir,
            'unet_lr' : self.unet_lr,
            'xformers' : self.xformers
        })
        return response
