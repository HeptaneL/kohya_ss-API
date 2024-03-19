import boto3
from configparser import ConfigParser

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
