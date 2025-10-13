class AnyClass():
    def __init__(self):
        pass

def dict_to_object(d):

    obj = AnyClass()

    for k, v in d.items():
        setattr(obj, k, v)
    
    return obj

def get_model_family(args):
    if args.model_name.startswith("llava-hf/llava-1.5-7b-hf"):
        return "llava"
    elif args.model_name.startswith("Salesforce/instructblip"):
        return "instructblip"
    elif args.model_name.startswith("Qwen/Qwen2.5-VL-7B-Instruct"):
        return "qwen"
    elif args.model_name.startswith("llava-hf/llava-onevision-qwen2-7b-ov-hf"):
        return "llava-onevision"
    else:
        raise Exception(f"`{args.model_name}` is not supported yet.")


# change these default arguments to change the templates for each part of the prompt 
def get_prompt_template_args(args):

    args.answer_template = "Answer:"
    args.further_instruction = "Answer the question using a single word or phrase."

    if args.dataset in ["cifar10", "cifar100", "imagenet100", "Pascal"]:
        args.candidates_template = "Select from the following classes: "

        args.caption_template = [
                "This is an image of a {}. ",
                "This is a photo of a {}. ",
                "An image of a {}. ",
                "A photo of a {}. ",
                "This is a {}. ",
                "A {}. ",
            ]

        if args.modality_to_report == "image":
            args.explicit_helper_template="Ignoring the caption, "

            if args.is_explicit_helper == 0:
                args.question_template = [
                    "What is the class of the input image?",
                    "What is in the image?",
                ]
            elif args.is_explicit_helper == 1:
                args.question_template = [
                    "what is the class of the input image?",
                    "what is in the image?",
                ]
            
        elif args.modality_to_report == "text":
            args.explicit_helper_template="Ignoring the image, "

            if args.is_explicit_helper == 0:
                args.question_template = [
                    "What is mentioned by the caption?",
                    "What does the caption say?",
                    "What is the class indicated by the caption?",
                ]
            elif args.is_explicit_helper == 1:
                args.question_template = [
                    "what is mentioned by the caption?",
                    "what does the caption say?",
                    "what is the class indicated by the caption?",
                ]

    elif args.dataset == "CUB_color":
        args.candidates_template = "Select from the following colors: "
        
        args.caption_template = [
                "This is a {} bird. ",
                "This bird has a primary color of {}. "
            ]

        if args.modality_to_report == "image":
            args.explicit_helper_template="Ignoring the caption, "

            if args.is_explicit_helper == 0:
                args.question_template = [
                    "What is the primary color of the bird shown in the image?",
                    "What color is the bird in the image?"
                ]
            elif args.is_explicit_helper == 1:
                args.question_template = [
                    "what is the primary color of the bird shown in the image?",
                    "what color is the bird in the image?"
                ]
            
        elif args.modality_to_report == "text":
            args.explicit_helper_template="Ignoring the image, "

            if args.is_explicit_helper == 0:
                args.question_template = [
                    "What is the primary color of the bird described in the caption?",
                    "What color is the bird described in the caption?"
                ]
            elif args.is_explicit_helper == 1:
                args.question_template = [
                    "what is the primary color of the bird described in the caption?",
                    "what color is the bird described in the caption?"
                ]

    return args