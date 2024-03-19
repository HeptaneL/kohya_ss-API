from http.server import BaseHTTPRequestHandler, HTTPServer
import subprocess
import threading
from datetime import datetime
import json
import boto3
from configparser import ConfigParser

# Global variable to track subprocess execution status
subprocess_running = False
current_params = None

class PostData:
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



# Function to execute the subprocess
def run_subprocess(command):
    global subprocess_running
    subprocess_running = True

    # Replace the command below with your desired subprocess command
    subprocess.run(command)

    subprocess_running = False

def upload_file_to_s3(file_path, object_name):
    # Load the AWS credentials from the configuration file
    config = ConfigParser()
    config.read('config.ini')
    access_key = config.get('aws', 'access_key')
    secret_key = config.get('aws', 'secret_key')
    bucket_name = config.get('aws', 'bucket_name')
    base_url = config.get('aws', 'base_url')

    # Create an S3 client with the provided credentials
    s3 = boto3.client('s3', aws_access_key_id=access_key, aws_secret_access_key=secret_key)

    try:
        s3.upload_file(file_path, bucket_name, object_name)
        url = f"{base_url}/{object_name}"
        print("File uploaded successfully.")
        return url, ""
    except Exception as e:
        print("Error uploading file:", str(e))
        return "", str(e)

def download_file_from_s3(bucket_name, object_name, file_path):
    s3 = boto3.client('s3')

    try:
        s3.download_file(bucket_name, object_name, file_path)
        print("File downloaded successfully.")
    except Exception as e:
        print("Error downloading file:", str(e))


# HTTP request handler
class RequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        global subprocess_running
        if self.path == '/param':
            # Return the current params as a response
            if current_params:
                response = current_params.get_json()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(response.encode())
            else:
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b'Params not available')
        else:
            # Return the subprocess status as a response
            response = json.dumps({"running_status": subprocess_running})
            self.send_response(200)
            self.end_headers()
            self.wfile.write(response.encode())

    def do_POST(self):
        global subprocess_running, current_params
        if self.path == '/model' and subprocess_running == False and current_params is not None:
            url, err = upload_file_to_s3(current_params.output_dir,current_params.output_name + "." + current_params.save_model_as)
            if url != "" :
                self.send_response(200, url)
                self.end_headers()
                self.wfile.write(b'Upload model')

            else:
                self.send_response(400, err)
                self.end_headers()
                self.wfile.write(b'Upload model failed')
        elif self.path == '/download-model':
            # Check for valid JSON content
            content_type = self.headers.get('Content-Type')
            if content_type != 'application/json':
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b'Invalid content type')
                return

            # Retrieve and parse the JSON data
            content_length = int(self.headers.get('Content-Length'))
            post_data = self.rfile.read(content_length)
            try:
                json_data = json.loads(post_data)
                download_file_from_s3(json_data.get('bucket_name'), json_data.get('object_name'), json_data.get('file_path'))
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'download lora')
            except json.JSONDecodeError:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b'Invalid JSON data')
                return

        # Check for valid JSON content
        content_type = self.headers.get('Content-Type')
        if content_type != 'application/json':
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b'Invalid content type')
            return

        # Retrieve and parse the JSON data
        content_length = int(self.headers.get('Content-Length'))
        post_data = self.rfile.read(content_length)
        try:
            json_data = json.loads(post_data)
        except json.JSONDecodeError:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b'Invalid JSON data')
            return

        # Create the PostData instance with parsed parameters
        params = PostData(
            bucket_no_upscale=json_data.get('bucket_no_upscale'),
            bucket_reso_steps=json_data.get('bucket_reso_steps'),
            cache_latents=json_data.get('cache_latents'),
            cache_latents_to_disk=json_data.get('cache_latents_to_disk'),
            enable_bucket=json_data.get('enable_bucket'),
            min_bucket_reso=json_data.get('min_bucket_reso'),
            max_bucket_reso=json_data.get('max_bucket_reso'),
            learning_rate=json_data.get('learning_rate'),
            logging_dir=json_data.get('logging_dir'),
            lr_scheduler=json_data.get('lr_scheduler'),
            lr_scheduler_num_cycles=json_data.get('lr_scheduler_num_cycles'),
            max_data_loader_n_workers=json_data.get('max_data_loader_n_workers'),
            max_grad_norm=json_data.get('max_grad_norm'),
            resolution=json_data.get('resolution'),
            max_train_steps=json_data.get('max_train_steps'),
            mixed_precision=json_data.get('mixed_precision'),
            network_alpha=json_data.get('network_alpha'),
            network_dim=json_data.get('network_dim'),
            network_module=json_data.get('network_module'),
            no_half_vae=json_data.get('no_half_vae'),
            optimizer_type=json_data.get('optimizer_type'),
            output_dir=json_data.get('output_dir'),
            output_name=json_data.get('output_name'),
            pretrained_model_name_or_path=json_data.get('pretrained_model_name_or_path'),
            save_every_n_epochs=json_data.get('save_every_n_epochs'),
            save_model_as=json_data.get('save_model_as'),
            save_precision=json_data.get('save_precision'),
            text_encoder_lr=json_data.get('text_encoder_lr'),
            train_batch_size=json_data.get('train_batch_size'),
            train_data_dir=json_data.get('train_data_dir'),
            unet_lr=json_data.get('unet_lr'),
            xformers=json_data.get('xformers')
        )
        current_params = params
        command = params.generate_command()
        # Trigger the subprocess execution
        if not subprocess_running:
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'Subprocess started')
            threading.Thread(target=run_subprocess, args=[command]).start()
        else:
            self.send_response(409)
            self.end_headers()
            self.wfile.write(b'Subprocess already running')
        

# Run the HTTP server
def run_server():
    server_address = ('', 8000)
    httpd = HTTPServer(server_address, RequestHandler)
    print('Server running on port 8000')
    httpd.serve_forever()

# Start the HTTP server in a separate thread
threading.Thread(target=run_server).start()
