import io
import json
import os
import uuid

from PIL import Image
import requests
import websocket


class ComfyUIClient:
    """
    Client for interacting with the ComfyUI server.
    
    Remember to change the server_address to match the IP address of your ComfyUI server.
    """
    def __init__(self, server_address='0.0.0.0:8888'):
        self.server_address = server_address
        self.client_id = str(uuid.uuid4())
        self.ws = None

    def establish_connection(self):
        """Establish a WebSocket connection to the ComfyUI server."""
        try:
            self.ws = websocket.WebSocket()
            self.ws.connect(f"ws://{self.server_address}/ws?clientId={self.client_id}")
            print('Connected to ComfyUI WebSocket server.')
        except Exception as e:
            print(f'Failed to connect to server: {e}')

    def queue_prompt(self, prompt_data):
        """Queue a prompt for image generation via API call."""
        url = f"http://{self.server_address}/prompt"
        payload = {
            "prompt": prompt_data,
            "client_id": self.client_id
        }
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f'Failed to queue prompt: {e}')
            return None

    def track_progress(self):
        """Track progress through WebSocket messages."""
        try:
            while True:
                message = self.ws.recv()
                data = json.loads(message)
                print(f'Progress: {data}')
                if data.get('type') == 'executing':
                    if data.get('data', {}).get('node') is None:
                        print('Image generation complete.')
                        break
        except Exception as e:
            print(f'Error while tracking progress: {e}')

    def fetch_generated_images(self, prompt_id):
        """Fetch generated images after completion."""
        url = f"http://{self.server_address}/history/{prompt_id}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            history_data = response.json()
            return history_data[prompt_id].get('outputs', {})
        except Exception as e:
            print(f'Failed to fetch generated images: {e}')
            return {}

    def save_images_locally(self, images, output_dir='output_images', media_type='gifs'):
        """Save fetched generated images locally.

        media_type can be 'images' or 'gifs'
        """
        os.makedirs(output_dir, exist_ok=True)

        for node_id, node_output in images.items():
            for image_data in node_output.get(media_type, []):
                image_url = f"http://{self.server_address}/view?filename={image_data['filename']}&subfolder={image_data['subfolder']}&type=output"
                try:
                    response = requests.get(image_url)
                    response.raise_for_status()

                    # Get file extension from filename
                    file_extension = os.path.splitext(image_data['filename'])[1].lower()
                    image_path = os.path.join(output_dir, image_data['filename'])

                    # Handle different file types
                    if file_extension in ['.gif', '.png', '.jpg', '.jpeg']:
                        image = Image.open(io.BytesIO(response.content))
                        image.save(image_path)
                    elif file_extension == '.mp4':
                        # Direct binary write for video files
                        with open(image_path, 'wb') as f:
                            f.write(response.content)
                    else:
                        print(f'Unsupported file type: {file_extension}')
                        continue

                    print(f'File saved to {image_path}')
                except Exception as e:
                    print(f'Failed to save file {image_data["filename"]}: {e}')


def run_comfyui_workflow(prompt_data, output_dir='output_images'):
    client = ComfyUIClient()
    client.establish_connection()
    
    queue_response = client.queue_prompt(prompt_data)
    if queue_response:
        prompt_id = queue_response['prompt_id']
        client.track_progress()
        images = client.fetch_generated_images(prompt_id)
        client.save_images_locally(images, output_dir)
    else:
        print("Failed to queue prompt. Workflow execution aborted.")

    client.ws.close()


def alter_mochi_v1_prompt(prompt: str = "Red panda eating bamboo with a waterfall behind."):
    workflow_json = json.loads("""{
  "1": {
    "inputs": {
      "prompt": "Model home tour of a luxury bath room.",
      "strength": 1,
      "force_offload": false,
      "clip": [
        "2",
        0
      ]
    },
    "class_type": "MochiTextEncode",
    "_meta": {
      "title": "Mochi TextEncode"
    }
  },
  "2": {
    "inputs": {
      "clip_name": "t5xxl_fp8_e4m3fn.safetensors",
      "type": "sd3"
    },
    "class_type": "CLIPLoader",
    "_meta": {
      "title": "Load CLIP"
    }
  },
  "4": {
    "inputs": {
      "model": "mochi_preview_dit_fp8_e4m3fn.safetensors",
      "vae": "mochi_preview_vae_decoder_bf16.safetensors",
      "precision": "fp8_e4m3fn",
      "attention_mode": "sage_attn",
      "cublas_ops": false,
      "rms_norm_func": "default",
      "trigger": [
        "8",
        0
      ]
    },
    "class_type": "DownloadAndLoadMochiModel",
    "_meta": {
      "title": "(Down)load Mochi Model"
    }
  },
  "8": {
    "inputs": {
      "prompt": "",
      "strength": 1,
      "force_offload": true,
      "clip": [
        "1",
        1
      ]
    },
    "class_type": "MochiTextEncode",
    "_meta": {
      "title": "Mochi TextEncode"
    }
  },
  "9": {
    "inputs": {
      "frame_rate": 24,
      "loop_count": 0,
      "filename_prefix": "Mochi_preview",
      "format": "video/h264-mp4",
      "pix_fmt": "yuv420p",
      "crf": 19,
      "save_metadata": true,
      "pingpong": false,
      "save_output": true,
      "images": [
        "11",
        0
      ]
    },
    "class_type": "VHS_VideoCombine",
    "_meta": {
      "title": "Video Combine 🎥🅥🅗🅢"
    }
  },
  "10": {
    "inputs": {
      "enable_vae_tiling": true,
      "auto_tile_size": false,
      "frame_batch_size": 10,
      "tile_sample_min_height": 160,
      "tile_sample_min_width": 312,
      "tile_overlap_factor_height": 0.25,
      "tile_overlap_factor_width": 0.25,
      "unnormalize": false,
      "vae": [
        "4",
        1
      ],
      "samples": [
        "14",
        0
      ]
    },
    "class_type": "MochiDecode",
    "_meta": {
      "title": "Mochi Decode"
    }
  },
  "11": {
    "inputs": {
      "image": [
        "10",
        0
      ]
    },
    "class_type": "GetImageSizeAndCount",
    "_meta": {
      "title": "Get Image Size & Count"
    }
  },
  "14": {
    "inputs": {
      "width": 848,
      "height": 480,
      "num_frames": 163,
      "steps": 50,
      "cfg": 4.5,
      "seed": 1,
      "model": [
        "4",
        0
      ],
      "positive": [
        "1",
        0
      ],
      "negative": [
        "8",
        0
      ]
    },
    "class_type": "MochiSampler",
    "_meta": {
      "title": "Mochi Sampler"
    }
  }
}""")
    workflow_json["1"]["inputs"]["prompt"] = prompt
    return workflow_json