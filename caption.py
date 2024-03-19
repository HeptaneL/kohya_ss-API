class Caption:
    def __init__(self,train_data_dir,caption_extension,batch_size,general_threshold,character_threshold,replace_underscores,model,recursive,max_data_loader_n_workers,debug,undesired_tags,frequency_tags,prefix,postfix,onnx,append_tags,force_download,caption_separator) -> None:
        self.train_data_dir = train_data_dir
        self.caption_extension = caption_extension
        self.batch_size = batch_size
        self.general_threshold = general_threshold
        self.character_threshold = character_threshold
        self.replace_underscores = replace_underscores
        self.model = model
        self.recursive = recursive
        self.max_data_loader_n_workers = max_data_loader_n_workers
        self.debug = debug
        self.undesired_tags = undesired_tags
        self.frequency_tags = frequency_tags
        self.prefix = prefix
        self.postfix = postfix
        self.onnx = onnx
        self.append_tags = append_tags
        self.force_download = force_download
        self.caption_separator = caption_separator

    def generate_command(self):
        command = [
            'accelerate',
            'launch',
            '"./finetune/tag_images_by_wd14_tagger.py"',
        ]

        return command
