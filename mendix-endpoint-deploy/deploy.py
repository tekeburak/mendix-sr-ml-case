import boto3
import os
from compress import compress_folder
from sagemaker.pytorch import PyTorchModel
from sagemaker import get_execution_role, Session
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer


def upload_file(model_path):
    """Upload a file to an S3 bucket

    :param model_path: PyTorch Model to upload
    :return: s3 URI
    """
    
    file_name = compress_folder(model_path)
    sess = Session()
    model_data = sess.upload_data(path=file_name, bucket=sess.default_bucket(), key_prefix="src")
    
    return model_data


role = "mendix"
model_folder = "model"

# Already uploaded model.tar.gz to S3 bucket
# model_data = upload_file(model_path=model_folder)
model_data = "s3://sagemaker-eu-north-1-450892103444/src/model.tar.gz"
print(f"MODEL S3 URI: {model_data}")


env = {
    'SAGEMAKER_REQUIREMENTS': 'requirements.txt', # path relative to `source_dir` below.
    'MMS_DEFAULT_RESPONSE_TIMEOUT': '500'
}

model = PyTorchModel(
    entry_point="inference.py",
    source_dir="model",
    role=role,
    model_data=model_data,
    framework_version="2.1",
    py_version="py310",
    env=env
)


# set local_mode to False if you want to deploy on a remote
# SageMaker instance

local_mode = True

if local_mode:
    instance_type = "local"
else:
    instance_type = "ml.g4dn.xlarge"

predictor = model.deploy(
    initial_instance_count=1,
    instance_type=instance_type,
    serializer=JSONSerializer(),
    deserializer=JSONDeserializer(),
)

print(f"ENDPOINT NAME: {predictor.endpoint_name}") # 'pytorch-inference-2023-12-20-07-43-25-455'