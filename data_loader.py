import os


class DataLoader:
    def __init__(self, base_dir):
        self.base_dir = base_dir

    def load_data(self):
        data = []
        for folder_name in os.listdir(self.base_dir):
            folder_path = os.path.join(self.base_dir, folder_name)
            if os.path.isdir(folder_path):
                for file_name in os.listdir(folder_path):
                    file_path = os.path.join(folder_path, file_name)
                    with open(file_path, 'r', encoding='utf-8') as file:
                        content = file.read()
                        document = {
                            'category': folder_name,
                            'content': content
                        }
                        data.append(document)
        return data
