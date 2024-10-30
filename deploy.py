"""Deploy OmniParse to AWS EC2 via Github action.

Usage:

    1. Create and populate the .env file:

        cat > .env <<EOF
AWS_ACCESS_KEY_ID=<your aws access key id>
AWS_SECRET_ACCESS_KEY=<your aws secret access key (required)>
AWS_REGION=<your aws region (required)>
GITHUB_OWNER=<your github owner (required)>  # e.g. microsoft
GITHUB_REPO=<your github repo (required)>    # e.g. OmniParse
GITHUB_TOKEN=<your github token (required)>
PROJECT_NAME=<your project name (required)>  # for tagging AWS resources
EOF

    2. Create a virtual environment for deployment:

        python3.10 -m venv venv
        source venv/bin/activate
        pip install -r deploy_requirements.txt

    3. Run the deployment script:

        python deploy.py start

    4. Wait for the build to succeed in Github actions (see console output for URL)

    5. Open the gradio interface (see console output for URL) and test it out.
       Note that it may take a minute for the interface to become available.
       You can also interact with the server programmatically:

        python client.py "http://<server_ip>:7861"

    6. Terminate the EC2 instance and stop incurring charges:

        python deploy.py stop

       Or, to shut it down without removing it:

        python deploy.py pause

       (This can later be re-started with the `start` command.)

    7. (optional) List all tagged instances with their respective statuses:

        python deploy.py status

    8. (optional) SSH into server:

        python deploy.py ssh

       Example commands:
            - View containers: `docker ps -a`
            - Tail logs: `docker logs -f <container_id>`
            - Enter container shell: `docker exec -it <container_id> /bin/bash`

Troubleshooting Token Scope Error:

    If you encounter an error similar to the following when pushing changes to
    GitHub Actions workflow files:

        ! [remote rejected] feat/docker -> feat/docker (refusing to allow a
        Personal Access Token to create or update workflow
        `.github/workflows/docker-build-ec2.yml` without `workflow` scope)

    This indicates that the Personal Access Token (PAT) being used does not
    have the necessary permissions ('workflow' scope) to create or update GitHub
    Actions workflows. To resolve this issue, you will need to create or update
    your PAT with the appropriate scope.

    Creating or Updating a Classic PAT with 'workflow' Scope:

    1. Go to GitHub and sign in to your account.
    2. Click on your profile picture in the top right corner, and then click 'Settings'.
    3. In the sidebar, click 'Developer settings'.
    4. Click 'Personal access tokens', then 'Classic tokens'.
    5. To update an existing token:
       a. Find the token you wish to update in the list and click on it.
       b. Scroll down to the 'Select scopes' section.
       c. Make sure the 'workflow' scope is checked. This scope allows for
          managing GitHub Actions workflows.
       d. Click 'Update token' at the bottom of the page.
    6. To create a new token:
       a. Click 'Generate new token'.
       b. Give your token a descriptive name under 'Note'.
       c. Scroll down to the 'Select scopes' section.
       d. Check the 'workflow' scope to allow managing GitHub Actions workflows.
       e. Optionally, select any other scopes needed for your project.
       f. Click 'Generate token' at the bottom of the page.
    7. Copy the generated token. Make sure to save it securely, as you will not
       be able to see it again.

    After creating or updating your PAT with the 'workflow' scope, update the
    Git remote configuration to use the new token, and try pushing your changes
    again.

    Note: Always keep your tokens secure and never share them publicly.

"""

import base64
import json
import os
import subprocess
import time

from botocore.exceptions import ClientError
from jinja2 import Environment, FileSystemLoader
from loguru import logger
from nacl import encoding, public
from pydantic_settings import BaseSettings
import boto3
import fire
import git
import paramiko
import requests

class Config(BaseSettings):
    AWS_ACCESS_KEY_ID: str
    AWS_SECRET_ACCESS_KEY: str
    AWS_REGION: str
    GITHUB_OWNER: str
    GITHUB_REPO: str
    GITHUB_TOKEN: str
    PROJECT_NAME: str

    AWS_EC2_AMI: str = "ami-0f9c346cdcac09fb5"  # Deep Learning AMI GPU PyTorch 2.0.1 (Ubuntu 20.04) 20230827
    AWS_EC2_DISK_SIZE: int = 100  # GB
    #AWS_EC2_INSTANCE_TYPE: str = "p3.2xlarge"  # (V100 16GB $3.06/hr x86_64)
    AWS_EC2_INSTANCE_TYPE: str = "g4dn.xlarge"  # (T4 16GB $0.526/hr x86_64)
    AWS_EC2_USER: str = "ubuntu"

    # Note: changing this requires changing the hard-coded value in other files
    PORT: int = 7861

    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'

    @property
    def AWS_EC2_KEY_NAME(self) -> str:
        return f"{self.PROJECT_NAME}-key"

    @property
    def AWS_EC2_KEY_PATH(self) -> str:
        return f"./{self.AWS_EC2_KEY_NAME}.pem"

    @property
    def AWS_EC2_SECURITY_GROUP(self) -> str:
        return f"{self.PROJECT_NAME}-SecurityGroup"

    @property
    def AWS_SSM_ROLE_NAME(self) -> str:
        return f"{self.PROJECT_NAME}-SSMRole"

    @property
    def AWS_SSM_PROFILE_NAME(self) -> str:
        return f"{self.PROJECT_NAME}-SSMInstanceProfile"

    @property
    def GITHUB_PATH(self) -> str:
        return f"{self.GITHUB_OWNER}/{self.GITHUB_REPO}"

config = Config()

def encrypt(public_key: str, secret_value: str) -> str:
    """
    Encrypts a Unicode string using the provided public key.

    Args:
        public_key (str): The public key for encryption, encoded in Base64.
        secret_value (str): The Unicode string to be encrypted.

    Returns:
        str: The encrypted value, encoded in Base64.
    """
    public_key = public.PublicKey(public_key.encode("utf-8"), encoding.Base64Encoder())
    sealed_box = public.SealedBox(public_key)
    encrypted = sealed_box.encrypt(secret_value.encode("utf-8"))
    return base64.b64encode(encrypted).decode("utf-8")

def set_github_secret(token: str, repo: str, secret_name: str, secret_value: str) -> None:
    """
    Sets a secret in the specified GitHub repository.

    Args:
        token (str): GitHub token with permissions to set secrets.
        repo (str): Repository path in the format "owner/repo".
        secret_name (str): The name of the secret to set.
        secret_value (str): The value of the secret.

    Returns:
        None
    """
    secret_value = secret_value or ""
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    response = requests.get(f"https://api.github.com/repos/{repo}/actions/secrets/public-key", headers=headers)
    response.raise_for_status()
    key = response.json()['key']
    key_id = response.json()['key_id']
    encrypted_value = encrypt(key, secret_value)
    secret_url = f"https://api.github.com/repos/{repo}/actions/secrets/{secret_name}"
    data = {"encrypted_value": encrypted_value, "key_id": key_id}
    response = requests.put(secret_url, headers=headers, json=data)
    response.raise_for_status()
    logger.info(f"set {secret_name=}")

def set_github_secrets() -> None:
    """
    Sets required AWS credentials and SSH private key as GitHub Secrets.

    Returns:
        None
    """
    # Set AWS secrets
    set_github_secret(config.GITHUB_TOKEN, config.GITHUB_PATH, 'AWS_ACCESS_KEY_ID', config.AWS_ACCESS_KEY_ID)
    set_github_secret(config.GITHUB_TOKEN, config.GITHUB_PATH, 'AWS_SECRET_ACCESS_KEY', config.AWS_SECRET_ACCESS_KEY)

    # Read the SSH private key from the file
    try:
        with open(config.AWS_EC2_KEY_PATH, 'r') as key_file:
            ssh_private_key = key_file.read()
        set_github_secret(config.GITHUB_TOKEN, config.GITHUB_PATH, 'SSH_PRIVATE_KEY', ssh_private_key)
    except IOError as e:
        logger.error(f"Error reading SSH private key file: {e}")

def create_key_pair(key_name: str = config.AWS_EC2_KEY_NAME, key_path: str = config.AWS_EC2_KEY_PATH) -> str | None:
    """
    Creates a new EC2 key pair and saves it to a file.

    Args:
        key_name (str): The name of the key pair to create. Defaults to config.AWS_EC2_KEY_NAME.
        key_path (str): The path where the key file should be saved. Defaults to config.AWS_EC2_KEY_PATH.

    Returns:
        str | None: The name of the created key pair or None if an error occurred.
    """
    ec2_client = boto3.client('ec2', region_name=config.AWS_REGION)
    try:
        key_pair = ec2_client.create_key_pair(KeyName=key_name)
        private_key = key_pair['KeyMaterial']

        # Save the private key to a file
        with open(key_path, "w") as key_file:
            key_file.write(private_key)
        os.chmod(key_path, 0o400)  # Set read-only permissions

        logger.info(f"Key pair {key_name} created and saved to {key_path}")
        return key_name
    except ClientError as e:
        logger.error(f"Error creating key pair: {e}")
        return None

def get_or_create_security_group_id(ports: list[int] = [22, config.PORT]) -> str | None:
    """
    Retrieves or creates a security group with the specified ports opened.

    Args:
        ports (list[int]): A list of ports to open in the security group. Defaults to [22, 7861].

    Returns:
        str | None: The ID of the security group, or None if an error occurred.
    """
    ec2 = boto3.client('ec2', region_name=config.AWS_REGION)

    # Construct ip_permissions list
    ip_permissions = [{
        'IpProtocol': 'tcp',
        'FromPort': port,
        'ToPort': port,
        'IpRanges': [{'CidrIp': '0.0.0.0/0'}]
    } for port in ports]

    try:
        response = ec2.describe_security_groups(GroupNames=[config.AWS_EC2_SECURITY_GROUP])
        security_group_id = response['SecurityGroups'][0]['GroupId']
        logger.info(f"Security group '{config.AWS_EC2_SECURITY_GROUP}' already exists: {security_group_id}")

        for ip_permission in ip_permissions:
            try:
                ec2.authorize_security_group_ingress(
                    GroupId=security_group_id,
                    IpPermissions=[ip_permission]
                )
                logger.info(f"Added inbound rule to allow TCP traffic on port {ip_permission['FromPort']} from any IP")
            except ClientError as e:
                if e.response['Error']['Code'] == 'InvalidPermission.Duplicate':
                    logger.info(f"Rule for port {ip_permission['FromPort']} already exists")
                else:
                    logger.error(f"Error adding rule for port {ip_permission['FromPort']}: {e}")

        return security_group_id
    except ClientError as e:
        if e.response['Error']['Code'] == 'InvalidGroup.NotFound':
            try:
                # Create the security group
                response = ec2.create_security_group(
                    GroupName=config.AWS_EC2_SECURITY_GROUP,
                    Description='Security group for specified port access',
                    TagSpecifications=[
                        {
                            'ResourceType': 'security-group',
                            'Tags': [{'Key': 'Name', 'Value': config.PROJECT_NAME}]
                        }
                    ]
                )
                security_group_id = response['GroupId']
                logger.info(f"Created security group '{config.AWS_EC2_SECURITY_GROUP}' with ID: {security_group_id}")

                # Add rules for the given ports
                ec2.authorize_security_group_ingress(GroupId=security_group_id, IpPermissions=ip_permissions)
                logger.info(f"Added inbound rules to allow access on {ports=}")

                return security_group_id
            except ClientError as e:
                logger.error(f"Error creating security group: {e}")
                return None
        else:
            logger.error(f"Error describing security groups: {e}")
            return None

def deploy_ec2_instance(
    ami: str = config.AWS_EC2_AMI,
    instance_type: str = config.AWS_EC2_INSTANCE_TYPE,
    project_name: str = config.PROJECT_NAME,
    key_name: str = config.AWS_EC2_KEY_NAME,
    disk_size: int = config.AWS_EC2_DISK_SIZE,
) -> tuple[str | None, str | None]:
    """
    Deploys an EC2 instance with the specified parameters.

    Args:
        ami (str): The Amazon Machine Image ID to use for the instance. Defaults to config.AWS_EC2_AMI.
        instance_type (str): The type of instance to deploy. Defaults to config.AWS_EC2_INSTANCE_TYPE.
        project_name (str): The project name, used for tagging the instance. Defaults to config.PROJECT_NAME.
        key_name (str): The name of the key pair to use for the instance. Defaults to config.AWS_EC2_KEY_NAME.
        disk_size (int): The size of the disk in GB. Defaults to config.AWS_EC2_DISK_SIZE.

    Returns:
        tuple[str | None, str | None]: A tuple containing the instance ID and IP address, or None, None if deployment fails.
    """
    ec2 = boto3.resource('ec2')
    ec2_client = boto3.client('ec2')

    # Check if key pair exists, if not create one
    try:
        ec2_client.describe_key_pairs(KeyNames=[key_name])
    except ClientError as e:
        create_key_pair(key_name)

    # Fetch the security group ID
    security_group_id = get_or_create_security_group_id()
    if not security_group_id:
        logger.error("Unable to retrieve security group ID. Instance deployment aborted.")
        return None, None

    # Check for existing instances
    instances = ec2.instances.filter(
        Filters=[
            {'Name': 'tag:Name', 'Values': [config.PROJECT_NAME]},
            {'Name': 'instance-state-name', 'Values': ['running', 'pending', 'stopped']}
        ]
    )

    for instance in instances:
        if instance.state['Name'] == 'running':
            logger.info(f"Instance already running: ID - {instance.id}, IP - {instance.public_ip_address}")
            return instance.id, instance.public_ip_address
        elif instance.state['Name'] == 'stopped':
            logger.info(f"Starting existing stopped instance: ID - {instance.id}")
            ec2_client.start_instances(InstanceIds=[instance.id])
            instance.wait_until_running()
            instance.reload()
            logger.info(f"Instance started: ID - {instance.id}, IP - {instance.public_ip_address}")
            return instance.id, instance.public_ip_address
        elif state == 'pending':
            logger.info(f"Instance is pending: ID - {instance.id}. Waiting for 'running' state.")
            try:
                instance.wait_until_running()  # Wait for the instance to be in 'running' state
                instance.reload()  # Reload the instance attributes
                logger.info(f"Instance is now running: ID - {instance.id}, IP - {instance.public_ip_address}")
                return instance.id, instance.public_ip_address
            except botocore.exceptions.WaiterError as e:
                logger.error(f"Error waiting for instance to run: {e}")
                return None, None
    # Define EBS volume configuration
    ebs_config = {
        'DeviceName': '/dev/sda1',  # You may need to change this depending on the instance type and AMI
        'Ebs': {
            'VolumeSize': disk_size,
            'VolumeType': 'gp3',  # Or other volume types like gp2, io1, etc.
            'DeleteOnTermination': True  # Set to False if you want to keep the volume after instance termination
        },
    }

    # Create a new instance if none exist
    new_instance = ec2.create_instances(
        ImageId=ami,
        MinCount=1,
        MaxCount=1,
        InstanceType=instance_type,
        KeyName=key_name,
        SecurityGroupIds=[security_group_id],
        BlockDeviceMappings=[ebs_config],
        TagSpecifications=[
            {
                'ResourceType': 'instance',
                'Tags': [{'Key': 'Name', 'Value': project_name}]
            },
        ]
    )[0]

    new_instance.wait_until_running()
    new_instance.reload()
    logger.info(f"New instance created: ID - {new_instance.id}, IP - {new_instance.public_ip_address}")
    return new_instance.id, new_instance.public_ip_address

def configure_ec2_instance(
    instance_id: str | None = None,
    instance_ip: str | None = None,
    max_ssh_retries: int = 10,
    ssh_retry_delay: int = 10,
    max_cmd_retries: int = 10,
    cmd_retry_delay: int = 30,
) -> tuple[str | None, str | None]:
    """
    Configures the specified EC2 instance for Docker builds.

    Args:
        instance_id (str | None): The ID of the instance to configure. If None, a new instance will be deployed. Defaults to None.
        instance_ip (str | None): The IP address of the instance. Must be provided if instance_id is manually passed. Defaults to None.
        max_ssh_retries (int): Maximum number of SSH connection retries. Defaults to 10.
        ssh_retry_delay (int): Delay between SSH connection retries in seconds. Defaults to 10.
        max_cmd_retries (int): Maximum number of command execution retries. Defaults to 10.
        cmd_retry_delay (int): Delay between command execution retries in seconds. Defaults to 30.

    Returns:
        tuple[str | None, str | None]: A tuple containing the instance ID and IP address, or None, None if configuration fails.
    """
    if not instance_id:
        ec2_instance_id, ec2_instance_ip = deploy_ec2_instance()
    else:
        ec2_instance_id = instance_id
        ec2_instance_ip = instance_ip  # Ensure instance IP is provided if instance_id is manually passed

    key = paramiko.RSAKey.from_private_key_file(config.AWS_EC2_KEY_PATH)
    ssh_client = paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    ssh_retries = 0
    while ssh_retries < max_ssh_retries:
        try:
            ssh_client.connect(hostname=ec2_instance_ip, username=config.AWS_EC2_USER, pkey=key)
            break  # Successful SSH connection, break out of the loop
        except Exception as e:
            ssh_retries += 1
            logger.error(f"SSH connection attempt {ssh_retries} failed: {e}")
            if ssh_retries < max_ssh_retries:
                logger.info(f"Retrying SSH connection in {ssh_retry_delay} seconds...")
                time.sleep(ssh_retry_delay)
            else:
                logger.error("Maximum SSH connection attempts reached. Aborting.")
                return

    # Setup commands to configure Docker, clean up unused resources, and handle name conflicts
    commands = [
        "sudo apt-get update",
        "sudo apt-get install -y docker.io",
        "sudo systemctl start docker",
        "sudo systemctl enable docker",
        "sudo usermod -a -G docker ${USER}",
        # Install Docker Compose
        "sudo curl -L \"https://github.com/docker/compose/releases/download/1.29.2/docker-compose-$(uname -s)-$(uname -m)\" -o /usr/local/bin/docker-compose",
        "sudo chmod +x /usr/local/bin/docker-compose",
        "sudo ln -s /usr/local/bin/docker-compose /usr/bin/docker-compose",
        # Docker cleanup and conflict resolution commands
        "sudo docker system prune -af --volumes",  # Clean up unused Docker resources
        "sudo docker rm -f omniparser-container || true",  # Remove conflicting container if it exists
    ]


    for command in commands:
        logger.info(f"Executing command: {command}")
        cmd_retries = 0
        while cmd_retries < max_cmd_retries:
            stdin, stdout, stderr = ssh_client.exec_command(command)
            exit_status = stdout.channel.recv_exit_status()  # Blocking call

            if exit_status == 0:
                logger.info(f"Command executed successfully")
                break
            else:
                error_message = stderr.read()
                if "Could not get lock" in str(error_message):
                    cmd_retries += 1
                    logger.warning(f"dpkg is locked, retrying command in {cmd_retry_delay} seconds... Attempt {cmd_retries}/{max_cmd_retries}")
                    time.sleep(cmd_retry_delay)
                else:
                    logger.error(f"Error in command: {command}, Exit Status: {exit_status}, Error: {error_message}")
                    break  # Non-dpkg lock error, break out of the loop

    ssh_client.close()
    return ec2_instance_id, ec2_instance_ip

def generate_github_actions_workflow() -> None:
    """
    Generates and writes the GitHub Actions workflow file for Docker build on EC2.

    Returns:
        None
    """
    current_branch = get_current_git_branch()

    _, host = deploy_ec2_instance()

    # Set up Jinja2 environment
    env = Environment(loader=FileSystemLoader('.'))
    template = env.get_template('docker-build-ec2.yml.j2')

    # Render the template with the current branch
    rendered_workflow = template.render(
        branch_name=current_branch,
        host=host,
        username=config.AWS_EC2_USER,
        project_name=config.PROJECT_NAME,
        github_path=config.GITHUB_PATH,
        github_repo=config.GITHUB_REPO,
    )

    # Write the rendered workflow to a file
    workflows_dir = '.github/workflows'
    os.makedirs(workflows_dir, exist_ok=True)
    with open(os.path.join(workflows_dir, 'docker-build-ec2.yml'), 'w') as file:
        file.write("# Autogenerated via deploy.py, do not edit!\n\n")
        file.write(rendered_workflow)
    logger.info("GitHub Actions EC2 workflow file generated successfully.")

def get_current_git_branch() -> str:
    """
    Retrieves the current active git branch name.

    Returns:
        str: The name of the current git branch.
    """
    repo = git.Repo(search_parent_directories=True)
    branch = repo.active_branch.name
    return branch

def get_github_actions_url() -> str:
    """
    Get the GitHub Actions URL for the user's repository.

    Returns:
        str: The Github Actions URL
    """
    url = f"https://github.com/{config.GITHUB_OWNER}/{config.GITHUB_REPO}/actions"
    return url

def get_gradio_server_url(ip_address: str) -> str:
    """
    Get the Gradio server URL using the provided IP address.

    Args:
        ip_address (str): The IP address of the EC2 instance running the Gradio server.

    Returns:
        str: The Gradio server URL
    """
    url = f"http://{ip_address}:{config.PORT}"
    return url

def git_push_set_upstream(branch_name: str) -> None:
    """
    Pushes the current branch to the remote 'origin' and sets it to track the upstream branch.
    If the push fails due to the branch being behind, pulls changes and retries.

    Args:
        branch_name (str): The name of the current branch to push.
    """
    try:
        # Push the current branch and set the remote 'origin' as upstream
        subprocess.run(["git", "push", "--set-upstream", "origin", branch_name], check=True)
        logger.info(f"Branch '{branch_name}' pushed and set up to track 'origin/{branch_name}'.")
    except subprocess.CalledProcessError as e:
        if "non-fast-forward" in str(e):
            logger.info("Branch is behind 'origin'. Attempting to pull and re-push...")
            try:
                subprocess.run(["git", "pull", "--rebase", "origin", branch_name], check=True)
                subprocess.run(["git", "push", "--set-upstream", "origin", branch_name], check=True)
                logger.info(f"Branch '{branch_name}' pushed after rebase.")
            except subprocess.CalledProcessError as pull_push_error:
                logger.error(f"Failed to push branch '{branch_name}' to 'origin' after pull: {pull_push_error}")
        else:
            logger.error(f"Failed to push branch '{branch_name}' to 'origin': {e}")

def update_git_remote_with_pat(github_owner: str, repo_name: str, pat: str):
    """
    Updates the git remote 'origin' to include the Personal Access Token in the URL.

    Args:
        github_owner (str): GitHub repository owner.
        repo_name (str): GitHub repository name.
        pat (str): Personal Access Token with the necessary scopes.

    """
    new_origin_url = f"https://{github_owner}:{pat}@github.com/{github_owner}/{repo_name}.git"
    try:
        # Remove the existing 'origin' remote
        subprocess.run(["git", "remote", "remove", "origin"], check=True)
        # Add the new 'origin' with the PAT in the URL
        subprocess.run(["git", "remote", "add", "origin", new_origin_url], check=True)
        logger.info("Git remote 'origin' updated successfully.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to update git remote 'origin': {e}")

class Deploy:

    @staticmethod
    def start() -> None:
        """
        Main method to execute the deployment process.

        Returns:
            None
        """
        set_github_secrets()
        instance_id, instance_ip = configure_ec2_instance()
        assert instance_ip, f"invalid {instance_ip=}"
        generate_github_actions_workflow()

        # Update the Git remote configuration to include the PAT
        update_git_remote_with_pat(
            config.GITHUB_OWNER, config.GITHUB_REPO, config.GITHUB_TOKEN,
        )

        # Add, commit, and push the workflow file changes, setting the upstream branch
        try:
            # Stage the workflow file
            subprocess.run(["git", "add", ".github/workflows/docker-build-ec2.yml"], check=True)

            # Check if there are any staged changes to commit
            result = subprocess.run(["git", "diff", "--cached", "--exit-code"])
            if result.returncode != 0:
                # Proceed with the commit only if there are changes
                subprocess.run(["git", "commit", "-m", "add workflow file"], check=True)
                logger.info("Changes committed successfully.")
            else:
                logger.info("No changes to commit.")

            current_branch = get_current_git_branch()
            git_push_set_upstream(current_branch)

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to commit or push changes: {e}")

        github_actions_url = get_github_actions_url()
        gradio_server_url = get_gradio_server_url(instance_ip)
        logger.info("Deployment process completed.")
        logger.info(f"Check the GitHub Actions at {github_actions_url}.")
        logger.info("Once the action is complete, run:")
        logger.info(f"    python client.py {gradio_server_url} <path>/<to>/<image.png>")

    @staticmethod
    def pause(project_name: str = config.PROJECT_NAME) -> None:
        """
        Shuts down the EC2 instance associated with the specified project name.

        Args:
            project_name (str): The project name used to tag the instance. Defaults to config.PROJECT_NAME.

        Returns:
            None
        """
        ec2 = boto3.resource('ec2')

        instances = ec2.instances.filter(
            Filters=[
                {'Name': 'tag:Name', 'Values': [project_name]},
                {'Name': 'instance-state-name', 'Values': ['running']}
            ]
        )

        for instance in instances:
            logger.info(f"Shutting down instance: ID - {instance.id}")
            instance.stop()

    @staticmethod
    def stop(
        project_name: str = config.PROJECT_NAME,
        security_group_name: str = config.AWS_EC2_SECURITY_GROUP,
    ) -> None:
        """
        Terminates the EC2 instance and deletes the associated security group.

        Args:
            project_name (str): The project name used to tag the instance. Defaults to config.PROJECT_NAME.
            security_group_name (str): The name of the security group to delete. Defaults to config.AWS_EC2_SECURITY_GROUP.

        Returns:
            None
        """
        ec2_resource = boto3.resource('ec2')
        ec2_client = boto3.client('ec2')

        # Terminate EC2 instances
        instances = ec2_resource.instances.filter(
            Filters=[
                {'Name': 'tag:Name', 'Values': [project_name]},
                {'Name': 'instance-state-name', 'Values': ['pending', 'running', 'shutting-down', 'stopped', 'stopping']}
            ]
        )

        for instance in instances:
            logger.info(f"Terminating instance: ID - {instance.id}")
            instance.terminate()
            instance.wait_until_terminated()
            logger.info(f"Instance {instance.id} terminated successfully.")

        # Delete security group
        try:
            ec2_client.delete_security_group(GroupName=security_group_name)
            logger.info(f"Deleted security group: {security_group_name}")
        except ClientError as e:
            if e.response['Error']['Code'] == 'InvalidGroup.NotFound':
                logger.info(f"Security group {security_group_name} does not exist or already deleted.")
            else:
                logger.error(f"Error deleting security group: {e}")

    @staticmethod
    def status() -> None:
        """
        Lists all EC2 instances tagged with the project name, along with their HTTP URLs.

        Returns:
            None
        """
        ec2 = boto3.resource('ec2')
        instances = ec2.instances.filter(
            Filters=[{'Name': 'tag:Name', 'Values': [config.PROJECT_NAME]}]
        )

        for instance in instances:
            public_ip = instance.public_ip_address
            if public_ip:
                http_url = f"http://{public_ip}:{config.PORT}"
                logger.info(f"Instance ID: {instance.id}, State: {instance.state['Name']}, HTTP URL: {http_url}")
            else:
                logger.info(f"Instance ID: {instance.id}, State: {instance.state['Name']}, HTTP URL: Not available (no public IP)")

    @staticmethod
    def ssh(project_name: str = config.PROJECT_NAME) -> None:
        """
        Establishes an SSH connection to the EC2 instance associated with the specified project name using subprocess.

        Args:
            project_name (str): The project name used to tag the instance. Defaults to config.PROJECT_NAME.

        Returns:
            None
        """
        ec2 = boto3.resource('ec2')
        instances = ec2.instances.filter(
            Filters=[
                {'Name': 'tag:Name', 'Values': [project_name]},
                {'Name': 'instance-state-name', 'Values': ['running']}
            ]
        )

        for instance in instances:
            logger.info(f"Attempting to SSH into instance: ID - {instance.id}, IP - {instance.public_ip_address}")

            # Build the SSH command
            ssh_command = [
                "ssh",
                "-i", config.AWS_EC2_KEY_PATH,
                f"{config.AWS_EC2_USER}@{instance.public_ip_address}"
            ]

            # Start an interactive shell session
            try:
                subprocess.run(ssh_command, check=True)
            except subprocess.CalledProcessError as e:
                logger.error(f"SSH connection failed: {e}")

if __name__ == "__main__":
    fire.Fire(Deploy)
