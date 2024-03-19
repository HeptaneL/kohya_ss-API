from http.server import BaseHTTPRequestHandler, HTTPServer
import subprocess
import threading
import json
from train_lora import TrainLora
from caption import Caption
from aws_s3 import upload_file_to_s3, download_file_from_s3

# Global variable to track subprocess execution status
subprocess_running = False
current_params = None


# Function to execute the subprocess
def run_subprocess(command):
    global subprocess_running
    subprocess_running = True
    # Replace the command below with your desired subprocess command
    subprocess.run(command)
    subprocess_running = False

# HTTP request handler
class RequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        global subprocess_running
        if self.path == '/param':
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
        elif self.path == '/caption':
            pass

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

        # Create the TrainLora instance with parsed parameters
        params = TrainLora(
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
