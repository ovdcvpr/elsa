import pandas as pd
from mapping import label2cat
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
import cv2
from torchvision import transforms
from PIL import Image
import PIL
import torch

IGNORE_CLASS = ["model_hint"]

class ClassManager(Dataset):

    def __init__(self, label_dir, image_folder, mode="CSA", th=1, res=256):
        """

        :param label_dir: directory with local annotations such as label_combo_freq.csv and label_ids.txt
        :param image_folder: directory with the images
        :param mode: CS filters out activities and others from ground truth, CSA filters others and removes CS only labels
                     CSAT keeps only annotations with all the categories
        :param th: min amount of occurrences needed to use that combo_label (currently ineffective)
        :param res: image resolution from the dataloader
        """
        assert mode in ["CS", "CSA", "CSAT"], "invalid mode submitted"
        assert th > 0

        self.mode = mode
        self.th = th
        self.label_dir = label_dir
        self.image_folder = image_folder

        normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        self.transform = transforms.Compose([
            transforms.Resize((res, res), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            # normalize,
        ])

        self.id2label = {}
        self.labels2id = {}
        self.prompts = []

        # the followings map series of class IDs to one single ID
        # this is useful so that the dataloader returns a single ID for each unique
        # combination of classes available. the class ids are sorted before being converted
        # to avoid different hyperID for the same combination of ids
        self.id2hyper_id = {}
        self.hyper_id2id = {}

        # id -> label
        with open(os.path.join(label_dir, "label_ids.txt"), 'r') as file:
            for line_number, line in enumerate(file, 0):
                class_name = line.strip()
                self.id2label[str(line_number)] = class_name
                self.labels2id[class_name] = str(line_number)


        ####
        # setting up the prompt list from all the available combinations allowed by 'mode'
        # also, remapping the combinations to a single unique ID so that the dataloader
        # returns one unique label instead of a variable amount of classes per bounding box
        label_combos = pd.read_csv(os.path.join(label_dir, "label_combo_freq.csv"))
        for idx, label_combo in label_combos.iterrows():
            class_names = self.parse_label(label_combo["Label Combination"])
            if class_names and class_names not in self.prompts:
                self.prompts.append(class_names)
                ids = [self.labels2id[lab] for lab in class_names]
                ids.sort(reverse=True)
                self.hyper_id2id[str(1000+idx)] = str(ids)
                self.id2hyper_id[str(ids)] = str(1000+idx)

        # loading actual annotations
        # filtering out images where no labels is valid with the current configuration
        self.image_links = pd.read_csv(os.path.join(label_dir, "images.csv"))
        raw_ann = pd.read_csv(os.path.join(label_dir, "label_per_box.csv"))
        print(f"Original number of bounding boxes: {len(raw_ann)}")
        raw_ann = raw_ann.groupby("ifile")
        self.ann = []
        for idx, group in raw_ann:
            tmp = []
            for _, bb in group.iterrows():
                parsed_lab = self.parse_label(bb["ilabel"])
                if parsed_lab:
                    bb["parsed"] = [self.labels2id[lab] for lab in parsed_lab]
                    bb["parsed"].sort(reverse=True)
                    bb["hyper_label"] = self.id2hyper_id[str(bb["parsed"])]
                    tmp.append(bb)
            if len(tmp) > 0:
                self.ann.append(tmp)
        print(f"New number of bounding boxes in {mode} mode: {self.check_total_len()}")

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, idx):
        """

        :param idx: id of sample
        :return:
        - image as a tensor
        - list of bounding boxes for the image in the format: x, y, w, h, label
        where label is a unique label assigned to the combination of multiple label
        this ensures supports for IoU in torchmetrics.
        the original combo labels can be retrieved using self.hyper_id2id
        then the original class names can be obtained with self.id2label
        """
        bbs = self.ann[idx]
        assert len(bbs) > 0
        filename = self.image_links.iloc[bbs[0]["ifile"]]["file"]
        img = PIL.Image.open(os.path.join(self.image_folder, filename)).convert('RGB')
        img = self.transform(img)
        data = [
            (bb.x, bb.y, bb.width, bb.height, eval(bb.hyper_label))
            for bb in bbs
        ]
        return img, torch.as_tensor(data)


    def check_total_len(self):
        """
        recursively count the number of bounding boxes
        """
        return sum([len(a) for a in self.ann])

    def parse_label(self, ilabel):
        """
        filters  what classes to use
        if CS -> filter out A and T and remove duplicates
        if CSA -> remove if A not there, filter out  T and remove duplicates
        if CSAT -> remove anything that has not the full series
        :param ilabel:
        :return:
        """
        class_id = eval(ilabel)  # e.g. [1, 5, 10, 11]
        
        # filtering out model hint
        class_id = [c_id for c_id in class_id if c_id != eval(self.labels2id["model_hint"])]
        
        class_names = [self.id2label[str(s)] for s in class_id]
        class_types = [label2cat[self.id2label[str(s)].lower()] for s in class_id]
        if self.mode == "CS":
            return [name for name, class_type in zip(class_names, class_types) if
                       class_type in ["condition", "state"]]
        elif self.mode == "CSA" and "activity" in class_types:
            return [name for name, class_type in zip(class_names, class_types) if
                        class_type in ["condition", "state", "activity"]]
        else:
            if "others" in class_types:
                return class_names
        return None

    def get_prompts(self):
        """
        return class names and label combo ids for the valid prompts in the current setup
        """
        return self.prompts, [[self.labels2id[a] for a in p] for p in self.prompts]

    def create_prompt(self, class_names):
        """
        generates a prompt given a set of class names, can be used in combination with get_prompts to
        generate all the possible prompts to consider that occurs at least once in the dataset
        """
        conditions = [a for a in class_names if label2cat[a.lower()] == "condition"]
        state = [a for a in class_names if label2cat[a.lower()] == "state"]
        activities = [a for a in class_names if label2cat[a.lower()] == "activity"]
        others = [a for a in class_names if label2cat[a.lower()] == "others"]
        # TODO: better merging into a prompt, check if lists are empty, "not sure" should probably be removed
        # TODO: use synonyms before merging?
        return ", ".join(conditions) + " " + \
               " , ".join(state) + " " + \
               ", ".join(activities) + " " + \
               " " + ", ".join(others)


if __name__ == '__main__':
    dataset = ClassManager(
        label_dir="gt_data/google",
        image_folder="/Users/REDACTED/Desktop/Labels_1K/google/coco/images",
        mode="CSA"
    )
    # the number of bounding boxes is variable, we need a custom collate or
    # batch size 1
    test = DataLoader(dataset, batch_size=1, shuffle=True)
    # check fetch
    _ = [_ for _ in test]

    # check prompts
    class_to_prompt, _ = dataset.get_prompts()
    print([dataset.create_prompt(classes) for classes in class_to_prompt])
    print("That's all folks!")