# tokenizer.py

class ByteTokenizer:
    def __init__(self):
        # Map bytes 0-255 to ids and reverse
        self.byte_to_id = {i: i for i in range(256)}
        self.id_to_byte = {i: i for i in range(256)}

    def encode(self, text):
        """
        Convert string to list of byte IDs
        """
        byte_array = text.encode("utf-8")
        return [self.byte_to_id[b] for b in byte_array]

    def decode(self, ids):
        """
        Convert list of byte IDs back to string
        """
        byte_array = bytes([self.id_to_byte[i] for i in ids])
        return byte_array.decode("utf-8")
