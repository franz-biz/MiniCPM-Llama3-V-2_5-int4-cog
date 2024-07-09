from cog import BasePredictor, Path, Input
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
import cv2

class Predictor(BasePredictor):

    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.model = AutoModel.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5-int4', trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5-int4', trust_remote_code=True)
        self.model.eval()

    def predict(self,
                input_media: Path = Input(
                    description="Path to the input image",
                ),
                prompt: str = Input(description="Image prompt"),
                temperature: float = Input(default=0.5,
                                         description="Temperature for the model",
                                         ge=0,
                                         le=1, )
                ) -> Path:
        image_path = str(input_media)
        cv2_image = cv2.imread(image_path)
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        # Convert NumPy array to PIL Image
        image = Image.fromarray(rgb_image)
        msgs = [{'role': 'user', 'content': prompt}]

        res = self.model.chat(
            image=image,
            msgs=msgs,
            tokenizer=self.tokenizer,
            sampling=True,  # if sampling=False, beam_search will be used by default
            temperature=float(temperature),
            # system_prompt='' # pass system_prompt if needed
        )
        return res
