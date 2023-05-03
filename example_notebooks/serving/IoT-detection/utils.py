"""
helper function

"""
def download_from_cloud(local_file_name, remote_file_name):
    """
    Upload a file to gcs bucket or s3 bucket.
    """
    import os
    
    cloud_name = remote_file_name.split('://')[0]
    if cloud_name =='gs':
        import gcsfs
        fs = gcsfs.GCSFileSystem(project=os.environ['GCP_PROJECT'])
    elif cloud_name =='s3':
        import s3fs
        fs = s3fs.S3FileSystem()
    else:
        raise NameError(f'cloud name {cloud_name} unknown')
    try:    
        print(f'Downloading from {remote_file_name} to {local_file_name}')
        fs.get(remote_file_name, local_file_name)
        print("done downloading!")
    except Exception as exp:
        print(f"Download failed: {exp}")
    return

def upload_to_cloud(local_file_name, remote_file_name):
    """
    Upload a file to gcs bucket or s3 bucket.
    """
    import os
    cloud_name = remote_file_name.split('://')[0]
    if cloud_name =='gs':
        import gcsfs
        fs = gcsfs.GCSFileSystem(project=os.environ['GCP_PROJECT'])
    elif cloud_name =='s3':
        import s3fs
        fs = s3fs.S3FileSystem()
    else:
        raise NameError(f'cloud name {cloud_name} unknown')
    try:    
        print(f'Uploading from {local_file_name} to {remote_file_name}')
        fs.put(local_file_name, remote_file_name)
        print("done uploading!")
    except Exception as exp:
        print(f"Uploading failed: {exp}")

    return fs