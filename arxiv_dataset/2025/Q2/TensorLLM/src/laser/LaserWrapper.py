from transformers import LlamaForCausalLM
from transformers import RobertaForMaskedLM
from transformers import GPTJForCausalLM

from laser.gptj_laser import GPTJLaser
from laser.llama2_laser import LLAMA2Laser
from laser.roberta_laser import RobertaLaser


class LaserWrapper:

    def __init__(self):
        pass
    
    @staticmethod
    def get_3D_Tucker_edited_model(model, lnum, device, qkvo_rank, attention_matrix, logger, in_place=True):
        # For ablation study purposes: Compressing W_Q, W_K, W_V, W_O separately vs. Compressing stacked QKVO tensor
        
        if type(model) == GPTJForCausalLM:
            
            logger.log("Editing a GPTJForCausalLM Model")
            return GPTJLaser.get_3D_Tucker_edited_model(model=model, 
                                                   lnum=lnum, 
                                                   device=device, 
                                                   qkvo_rank=qkvo_rank,
                                                   attention_matrix=attention_matrix,
                                                   in_place=in_place)
        
    @staticmethod
    def get_QKVO_edited_model(model, lnum, device, qkvo_rank, head_dim_rank=None, stack_rank=None, new_reshape=False, qkvo_intervention='partial_tucker', logger=None, in_place=True):
        if type(model) == GPTJForCausalLM:
            
            logger.log("Editing a GPTJForCausalLM Model")
            return GPTJLaser.get_QKVO_edited_model(model=model, 
                                                   lnum=lnum, 
                                                   device=device, 
                                                   qkvo_rank=qkvo_rank,
                                                   stack_rank=stack_rank,
                                                   head_dim_rank=head_dim_rank,
                                                   qkvo_intervention=qkvo_intervention,
                                                   in_place=in_place)

        elif type(model) == LlamaForCausalLM:
            
            logger.log("Editing a LlamaForCausalLM Model")
            return LLAMA2Laser.get_QKVO_edited_model(model=model, 
                                         lnum=lnum, 
                                         device=device, 
                                         qkvo_rank=qkvo_rank,
                                         head_dim_rank=head_dim_rank,
                                         stack_rank=stack_rank,
                                         qkvo_intervention=qkvo_intervention,
                                         in_place=in_place)
            
        elif type(model) == RobertaForMaskedLM:

            logger.log("Editing a RobertaForMaskedLM Model")
            return RobertaLaser.get_QKVO_edited_model(model=model, 
                                                   lnum=lnum, 
                                                   device=device, 
                                                   qkvo_rank=qkvo_rank,
                                                   stack_rank=stack_rank,
                                                   head_dim_rank=head_dim_rank,
                                                   qkvo_intervention=qkvo_intervention,
                                                   in_place=in_place)
                                         

    @staticmethod
    def get_edited_model(model, lname, lnum, rate, intervention="rank-reduction", logger=None, in_place=True):

        if type(model) == LlamaForCausalLM:
            logger.log("Editing a LlamaForCausalLM Model")

            return LLAMA2Laser.get_edited_model(model=model,
                                                lname=lname,
                                                lnum=lnum,
                                                rate=rate,
                                                intervention=intervention,
                                                logger=logger,
                                                in_place=in_place)

        elif type(model) == RobertaForMaskedLM:

            logger.log("Editing a RobertaForMaskedLM Model")
            return RobertaLaser.get_edited_model(model=model,
                                                 lname=lname,
                                                 lnum=lnum,
                                                 rate=rate,
                                                 intervention=intervention,
                                                 logger=logger,
                                                 in_place=in_place)

        elif type(model) == GPTJForCausalLM:

            logger.log("Editing a GPTJForCausalLM Model")
            return GPTJLaser.get_edited_model(model=model,
                                              lname=lname,
                                              lnum=lnum,
                                              rate=rate,
                                              intervention=intervention,
                                              logger=logger,
                                              in_place=in_place)

        else:
            raise AssertionError(f"Unhandled model of type {type(model)}.")
