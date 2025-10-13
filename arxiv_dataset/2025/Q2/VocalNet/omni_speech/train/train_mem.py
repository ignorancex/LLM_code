from omni_speech.train.train_tdm import train

if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
