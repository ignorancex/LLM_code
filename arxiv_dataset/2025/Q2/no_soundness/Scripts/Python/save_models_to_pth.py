import Models.models as models
import torch
import os

def main():

    model_32bit_adv = models.get_Wk17a_prec_32_adv()
    model_64bit_adv = models.get_Wk17a_prec_64_adv()

    model_p1 = models.get_Wk17a_order_pattern_1_adv()
    model_p2 = models.get_Wk17a_order_pattern_2_adv()
    model_p3 = models.get_Wk17a_order_pattern_3_adv()

    model_p1_f32 = models.get_Wk17a_order_pattern_1_f32_adv()
    model_p2_f32 = models.get_Wk17a_order_pattern_2_f32_adv()
    model_p3_f32 = models.get_Wk17a_order_pattern_3_f32_adv()

    model_order = models.get_Wk17a_order_bias_adv()

    torch.save(model_32bit_adv.state_dict(), os.path.join("..", "..", "Models", "wk17a_32bit_adversary.pth"))
    torch.save(model_64bit_adv.state_dict(), os.path.join("..", "..", "Models", "wk17a_64bit_adversary.pth"))

    torch.save(model_p1.state_dict(), os.path.join("..", "..", "Models", "wk17a_order_pattern_1_f64_adversary.pth"))
    torch.save(model_p2.state_dict(), os.path.join("..", "..", "Models", "wk17a_order_pattern_2_f64_adversary.pth"))
    torch.save(model_p3.state_dict(), os.path.join("..", "..", "Models", "wk17a_order_pattern_3_f64_adversary.pth"))

    torch.save(model_p1_f32.state_dict(), os.path.join("..", "..", "Models", "wk17a_order_pattern_1_f32_adversary.pth"))
    torch.save(model_p2_f32.state_dict(), os.path.join("..", "..", "Models", "wk17a_order_pattern_2_f32_adversary.pth"))
    torch.save(model_p3_f32.state_dict(), os.path.join("..", "..", "Models", "wk17a_order_pattern_3_f32_adversary.pth"))

    torch.save(model_order.state_dict(), os.path.join("..", "..", "Models", "wk17a_order_adversary.pth"))

if __name__ == "__main__":
    main()