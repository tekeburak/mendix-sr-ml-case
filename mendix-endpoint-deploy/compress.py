import tarfile
import os


def compress_model(model_path):
    output_tarfile = 'model.tar.gz'
    with tarfile.open(output_tarfile, 'w:gz') as tar:
        tar.add(model_path, arcname=os.path.basename(model_path))
    return output_tarfile


def compress_folder(folder_path):
    output_tarfile = 'model.tar.gz'
    with tarfile.open(output_tarfile, "w:gz") as tar:
        tar.add(folder_path, arcname=os.path.basename(folder_path))
    return output_tarfile
