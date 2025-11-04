import numpy as np
import torch
from tqdm import tqdm
import torch.nn.functional as F

def dpc(labels, outputs):
    """Calculate the Distinct Purchased Creatives (DPC) metric.
    Args:
        labels (torch.Tensor): The ground truth labels of shape (batch_size, seq_len).
        outputs (torch.Tensor): The model outputs of shape (batch_size, k, seq_len).
    Returns:
        tuple: A tuple containing:
            - set: Unique correct item sequences at k=1.
            - set: Unique correct item sequences at k=10.
            - set: Total unique predicted item sequences at k=1.
            - set: Total unique predicted item sequences at k=10.
    """
    # Get index of matched items where outputs[:, 0, 1:] == labels[:, 1:], add all item tensors labels[:, 1:] to matched_items set
    matched_items_1 = set()
    matched_indices_1 = torch.where(torch.all(outputs[:, 0, 1:] == labels[:, 1:], dim=1))[0]
    for idx in matched_indices_1:
        matched_items_1.add(tuple(labels[idx, 1:].cpu().numpy().tolist()))

    # At k=10
    matched_items_10 = set()
    matched_indices_10 = torch.where(torch.any(torch.all(outputs[:, :10, 1:] == labels[:, 1:].unsqueeze(1), dim=2), dim=1))[0]
    for idx in matched_indices_10:
        matched_items_10.add(tuple(labels[idx, 1:].cpu().numpy().tolist()))

    all_dpc_1 = set()
    all_dpc_10 = set()
    for i in range(outputs.shape[0]):
        all_dpc_1.add(tuple(outputs[i, 0, 1:].cpu().numpy().tolist()))
        for j in range(10):
            all_dpc_10.add(tuple(outputs[i, j, 1:].cpu().numpy().tolist()))

    return matched_items_1, matched_items_10, all_dpc_1, all_dpc_10



def dcg(scores):
    """Compute the Discounted Cumulative Gain."""
    scores = np.asarray(scores, dtype=np.float32)  # Ensure scores is an array of floats
    return np.sum((2**scores - 1) / np.log2(np.arange(2, scores.size + 2)))

def ndcg_at_k(r, k):
    """Compute NDCG at rank k."""
    r = np.asarray(r, dtype=np.float32)[:k]  # Ensure r is an array of floats and take top k scores
    dcg_max = dcg(sorted(r, reverse=True))
    if not dcg_max:
        return 0.
    return dcg(r) / dcg_max

def revenue_metrics(matches, conv_amount_value):
    """
    Calculate hit metrics given matches array and conversion amount.

    Hit: Prediction where full sequence matches ground truth.

    Args:
        matches: numpy array indicating which predictions match ground truth
        conv_amount_value: scalar conversion amount for this sample

    Returns:
        hit_at_1: bool, whether there's a hit at k=1
        hit_revenue_at_1: float, revenue at k=1 (conv_amount or 0)
        hit_at_10: bool, whether there's a hit at k=10
        hit_revenue_at_10: float, revenue at k=10 (conv_amount or 0)
    """

    # Hit at k=1
    hit_at_1 = matches[:1].sum() > 0
    hit_revenue_at_1 = conv_amount_value if hit_at_1 else 0.0

    # Hit at k=10
    hit_at_10 = matches[:10].sum() > 0
    hit_revenue_at_10 = conv_amount_value if hit_at_10 else 0.0

    return hit_at_1, hit_revenue_at_1, hit_at_10, hit_revenue_at_10

def calculate_metrics(outputs, labels, conv_amount=None, eval_mode='Target'):
    batch_size, k, _ = outputs.shape  # Assuming outputs is [batch_size, 10, seq_len]
    recall_at_1, recall_at_5, recall_at_10 = [], [], []
    ndcg_at_5, ndcg_at_10 = [], []

    # Only compute hit metrics for Target mode
    if eval_mode == 'Target':
        hit_count_at_1 = 0
        hit_count_at_10 = 0
        total_hit_revenue_at_1 = 0.0
        total_hit_revenue_at_10 = 0.0

    for i in range(batch_size):
        label = labels[i].unsqueeze(0)  # [1, seq_len]
        out = outputs[i]  # [10, seq_len]

        matches = torch.all(torch.eq(out.unsqueeze(1), label.unsqueeze(0)), dim=2)  # [10, 1, seq_len] -> [10, 1]
        matches = matches.any(dim=1).cpu().numpy()  # [10]

        # Recall
        recall_at_1.append(matches[:1].sum() / 1.0)  # Assuming each label has only 1 correct match.
        recall_at_5.append(matches[:5].sum() / 1.0)  # Assuming each label has only 1 correct match.
        recall_at_10.append(matches.sum() / 1.0)

        # NDCG (binary relevance)
        ndcg_at_5.append(ndcg_at_k(matches, 5))
        ndcg_at_10.append(ndcg_at_k(matches, 10))

        # Hit metrics (only for Target mode)
        if eval_mode == 'Target':
            hit_at_1, hit_revenue_at_1, hit_at_10, hit_revenue_at_10 = revenue_metrics(
                matches, conv_amount[i].item()
            )

            matches, labels

            hit_count_at_1 += int(hit_at_1)
            hit_count_at_10 += int(hit_at_10)
            total_hit_revenue_at_1 += hit_revenue_at_1
            total_hit_revenue_at_10 += hit_revenue_at_10

    # Calculate mean metrics
    if eval_mode == 'Target':
        metrics = (
            np.mean(recall_at_1),
            np.mean(recall_at_5),
            np.mean(recall_at_10),
            np.mean(ndcg_at_5),
            np.mean(ndcg_at_10),
            hit_count_at_1,
            total_hit_revenue_at_1,
            hit_count_at_10,
            total_hit_revenue_at_10,
        )
    else:
        metrics = (
            np.mean(recall_at_1),
            np.mean(recall_at_5),
            np.mean(recall_at_10),
            np.mean(ndcg_at_5),
            np.mean(ndcg_at_10),
        )

    return metrics

def calculate_acc_recall_precision(outputs, labels):
    batch_size = outputs.shape[0]
    correct = (outputs == labels).sum().item()
    true_positives = ((outputs == 1) & (labels == 1)).sum().item()
    predicted_positives = (outputs == 1).sum().item()
    actual_positives = (labels == 1).sum().item()
    
    return correct, batch_size, true_positives, predicted_positives, actual_positives




def prepare_beam_search_inputs(model, input_ids, attention_mask, decoder_input_ids, batch_size, num_beams):
    """
    Adpated from huggingface's implementation
    https://github.com/huggingface/transformers/blob/v4.39.3/src/transformers/generation/utils.py#L2823
    
    Prepares and duplicates the input data for beam search decoding.

    This function initializes decoder input IDs and beam scores, creates an offset for beam indices, 
    and expands the input_ids and attention_mask tensors to accommodate the specified number of beams for each instance in the batch.

    Parameters:
    - model (torch.nn.Module): The model being used for beam search, which must have a 'config.decoder_start_token_id' attribute for initializing decoder input IDs.
    - input_ids (torch.Tensor): The input IDs tensor of shape (batch_size, sequence_length) used for the encoder part of the model.
    - attention_mask (torch.Tensor): The attention mask tensor of shape (batch_size, sequence_length) indicating to the model which tokens should be attended to.
    - batch_size (int): The number of instances per batch in the input data.
    - num_beams (int): The number of beams to use in beam search. This expands the input data and scores accordingly.

    Returns:
    - input_ids (torch.Tensor): The expanded input IDs tensor to match the number of beams, shape (batch_size * num_beams, sequence_length).
    - attention_mask (torch.Tensor): The expanded attention mask tensor to match the number of beams, shape (batch_size * num_beams, sequence_length).
    - initial_decoder_input_ids (torch.Tensor): The initialized decoder input IDs for each beam, shape (batch_size * num_beams, 1).
    - initial_beam_scores (torch.Tensor): The initialized scores for each beam, flattened to a single dimension, shape (batch_size * num_beams,).
    - beam_idx_offset (torch.Tensor): An offset for each beam index to assist in reordering beams during the search, shape (batch_size * num_beams,).

    Each input sequence is replicated 'num_beams' times to provide separate candidate paths in beam search. Beam scores are initialized with 0 for the first beam and a very low number (-1e9) for others to ensure the first token of each sequence is chosen from the first beam.
    """
    beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
    beam_scores[:, 1:] = -1e9  # Set a low score for all but the first beam to ensure the first beam is selected initially
    initial_beam_scores = beam_scores.view((batch_size * num_beams,))

    beam_idx_offset = torch.arange(batch_size, device=model.device).repeat_interleave(num_beams) * num_beams
    initial_decoder_input_ids = decoder_input_ids.repeat_interleave(num_beams, dim=0)
    input_ids = input_ids.repeat_interleave(num_beams, dim=0)
    attention_mask = attention_mask.repeat_interleave(num_beams, dim=0)

    return input_ids, attention_mask, initial_decoder_input_ids, initial_beam_scores, beam_idx_offset


def beam_search_step(logits, decoder_input_ids, beam_scores, beam_idx_offset, batch_size, num_beams):
    """
    Adpated from huggingface's implementation
    https://github.com/huggingface/transformers/blob/v4.39.3/src/transformers/generation/utils.py#L2823

    Executes one step of beam search, calculating the next set of input IDs based on logits from a model.

    This function expands the current beam, calculates scores for all possible next tokens, selects the top tokens for each beam, and prepares the input IDs for the next iteration of the model. It utilizes logits output by the model to determine the most likely next tokens and updates the beam scores.

    Parameters:
    - logits (torch.Tensor): Logits returned from the model, shape (batch_size * num_beams, sequence_length, vocab_size).
    - decoder_input_ids (torch.Tensor): Current decoder input IDs, shape (batch_size * num_beams, current_sequence_length).
    - beam_scores (torch.Tensor): Current scores for each beam, shape (batch_size * num_beams,).
    - beam_idx_offset (torch.Tensor): Index offsets for each beam to handle batches correctly, shape (batch_size * num_beams,).
    - batch_size (int): Number of sequences being processed in a batch.
    - num_beams (int): Number of beams used in the beam search.

    Returns:
    - decoder_input_ids (torch.Tensor): Updated decoder input IDs after adding the next tokens, shape (batch_size * num_beams, current_sequence_length + 1).
    - beam_scores (torch.Tensor): Updated scores for each beam, shape (batch_size * num_beams,).

    The function selects the top `2 * num_beams` tokens from the logits based on their scores, reshapes and adjusts them based on the existing beam scores, and determines the next tokens to add to each beam path. The updated paths are then returned for use in the next iteration of the beam search.
    """
    assert batch_size * num_beams == logits.shape[0]
    
    vocab_size = logits.shape[-1]
    next_token_logits = logits[:, -1, :]
    next_token_scores = torch.log_softmax(next_token_logits, dim=-1)  # Calculate log softmax over the last dimension

    next_token_scores = next_token_scores + beam_scores[:, None].expand_as(next_token_scores)
    next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)
    next_token_scores, next_tokens = torch.topk(next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True)
    
    next_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor")
    next_tokens = next_tokens % vocab_size
    
    beam_scores = next_token_scores[:, :num_beams].reshape(-1)
    beam_next_tokens = next_tokens[:, :num_beams].reshape(-1)
    beam_idx = next_indices[:, :num_beams].reshape(-1)

    # beam_idx_offset: beam_idx contains sequence indicies relative to each individual batch. We need to offset the indicies to retrieve the correct sequence in the corresponding batch
    # for example, when batch_size = 2, beam_size = 3, beam_idx_offset = [0, 0, 0, 3, 3, 3]
    decoder_input_ids = torch.cat([decoder_input_ids[beam_idx + beam_idx_offset, :], beam_next_tokens.unsqueeze(-1)], dim=-1)

    return decoder_input_ids, beam_scores


def beam_search(model, input_ids, attention_mask, decoder_input_ids = None, max_length=6, num_beams=1, num_return_sequences=1, return_score=False, encoder_outputs = None):
    """
    Adpated from huggingface's implementation
    https://github.com/huggingface/transformers/blob/v4.39.3/src/transformers/generation/utils.py#L2823

    Perform beam search to generate sequences using the specified model. 
    
    *** This implementation does not include stopping conditions based on end-of-sequence (EOS) tokens. Instead, the
    sequence generation is controlled solely by the `max_length` parameter. ***

    Note: In scenarios where the generation should explicitly detect and respond to EOS tokens 
    to terminate the sequence early, this function would need modifications. In the current setup,
    setting `max_length` to a suitable fixed value (e.g., 6) can serve the purpose by limiting
    the maximum sequence length.

    Parameters:
    - model (torch.nn.Module): The model to use for generating sequences.
    - input_ids (torch.Tensor): Tensor of input ids.
    - attention_mask (torch.Tensor): Tensor representing the attention mask.
    - max_length (int): Maximum length of the sequence to be generated; controls when to stop extending the sequence. [BOS, behavior_token, item_token_1, item_token_2, item_token_3, EOS]
    - num_beams (int): Number of beams for beam search.
    - num_return_sequences (int): Number of sequences to return.
    - return_score (bool): If True, returns a tuple of (sequences, scores) where 'scores' are the average log likelihood of the returned sequences.

    Returns:
    - torch.Tensor: The final decoder input ids from the beam search, or a tuple of (decoder_input_ids, scores) if 'return_score' is True.

    Example usage:
    # Assuming the model, input_ids, and attention_mask are predefined:
    sequences = beam_search(model, input_ids, attention_mask, max_length=6, num_beams=5, num_return_sequences=5)
    """

    assert max_length <= 7, "This implementation does not include stopping conditions based on end-of-sequence (EOS) tokens. So setting max_length > 7 would be meaningless for TIGER."
    
    batch_size = input_ids.shape[0]
    
    # Prepare beam search inputs
    input_ids, attention_mask, decoder_input_ids, beam_scores, beam_idx_offset = prepare_beam_search_inputs(
        model, input_ids, attention_mask, decoder_input_ids, batch_size, num_beams
    )
    # Store encoder_outputs to prevent running full forward path repeatedly
    if encoder_outputs is None:
        with torch.no_grad():
            encoder_outputs = model.get_encoder()(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
    else:
        encoder_outputs = encoder_outputs.repeat_interleave(num_beams, dim = 0)

    # Beam search loop
    while decoder_input_ids.shape[1] < max_length:
        with torch.no_grad():
            outputs = model(encoder_outputs = encoder_outputs, attention_mask=attention_mask, decoder_input_ids = decoder_input_ids)
        
        decoder_input_ids, beam_scores = beam_search_step(outputs.logits, decoder_input_ids, beam_scores, beam_idx_offset, batch_size, num_beams)

    # (batch_size * num_beams, ) -> (batch_size * num_return_sequences, )
    selection_mask = torch.zeros(batch_size, num_beams, dtype=bool)
    selection_mask[:, :num_return_sequences] = True

    if return_score:
        return decoder_input_ids[selection_mask.view(-1), :], beam_scores[selection_mask.view(-1)] / (decoder_input_ids.shape[1] - 1)
        
    return decoder_input_ids[selection_mask.view(-1), :]
    

def evaluate(model, dataloader, device, item_len, num_beams=10, eval_mode = 'Target', no_output=False, behavior_token = True, reverse_bt = False):
    model.eval()
    recall_at_1s = []
    recall_at_5s = []
    recall_at_10s = []
    ndcg_at_5s = []
    ndcg_at_10s = []
    losses = []
    hit_count_at_1_list = []
    total_hit_revenue_at_1_list = []
    hit_count_at_10_list = []
    total_hit_revenue_at_10_list = []
    dpc_1_set, dpc_10_set, total_1_set, total_10_set = set(), set(), set(), set()
    if not no_output:
        progress_bar = tqdm(range(len(dataloader)))
    for batch in dataloader:
        batch_size = batch['input_ids'].shape[0]
        input_ids = batch['input_ids'].to(device).to(torch.long)
        attention_mask = batch['attention_mask'].to(device).to(torch.long)
        labels = batch['labels'].to(device).to(torch.long)
        conv_amount = batch['conv_amount'].to(device).to(torch.int32)
        label_len = labels.shape[1]
        if behavior_token and (not reverse_bt):
            if eval_mode == 'Target':
                decoder_input = torch.tensor([0, 1]).unsqueeze(0).repeat(batch_size, 1).to(device)
            elif eval_mode == 'Behavior_specific':
                decoder_input = torch.cat([torch.zeros(batch_size, 1, device=input_ids.device, dtype=torch.long), labels[:, :1]], dim=-1)
            elif eval_mode == 'Behavior_item':
                decoder_input = torch.zeros(batch_size, 1, device=input_ids.device, dtype=torch.long)
        else:
            decoder_input = torch.zeros(batch_size, 1, device=input_ids.device, dtype=torch.long)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            losses.append(outputs.loss.item())
            del(outputs)
        if behavior_token and (not reverse_bt):
            outputs = beam_search(model, input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input, max_length=label_len + 1, num_beams=num_beams, num_return_sequences=10)
        else:
            outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input, max_length=label_len + 1, num_beams=num_beams, num_return_sequences=10)
        outputs = outputs[:, 1:-1].reshape(batch_size, 10, -1)  # Remove BOS and EOS
        labels = labels[:,:-1] # Remove EOS

        batch_dpc_1, batch_dpc_10, batch_total_1, batch_total_10 = dpc(labels, outputs)
        dpc_1_set.update(batch_dpc_1)
        dpc_10_set.update(batch_dpc_10)
        total_1_set.update(batch_total_1)
        total_10_set.update(batch_total_10)
        # Add this for Behavior_only mode:
        if eval_mode == 'Behavior_only':
            outputs = outputs[:, :, :1]  # Keep only first token (behavior), the rest are item tokens
            labels = labels[:, :1] # Same

        if eval_mode == 'Target':
            recall_at_1, recall_at_5, recall_at_10, ndcg_at_5, ndcg_at_10, hit_count_at_1, total_hit_revenue_at_1, hit_count_at_10, total_hit_revenue_at_10 = calculate_metrics(outputs, labels, conv_amount, eval_mode)
            hit_count_at_1_list.append(hit_count_at_1)
            total_hit_revenue_at_1_list.append(total_hit_revenue_at_1)
            hit_count_at_10_list.append(hit_count_at_10)
            total_hit_revenue_at_10_list.append(total_hit_revenue_at_10)
        else:
            recall_at_1, recall_at_5, recall_at_10, ndcg_at_5, ndcg_at_10 = calculate_metrics(outputs, labels, eval_mode=eval_mode)
        recall_at_1s.append(recall_at_1)
        recall_at_5s.append(recall_at_5)
        recall_at_10s.append(recall_at_10)
        ndcg_at_5s.append(ndcg_at_5)
        ndcg_at_10s.append(ndcg_at_10)
        if not no_output:
            progress_bar.set_description(f"recall@10: {(sum(recall_at_10s) / len(recall_at_10s)):.4f}, NDCG@10: {(sum(ndcg_at_10s) / len(ndcg_at_10s)):.4f}")
            progress_bar.update(1)
    if not no_output:
        progress_bar.close()

    print(f"Validation Loss: {sum(losses) / len(losses)}")
    print(f"recall@1: {sum(recall_at_1s) / len(recall_at_1s)}")
    print(f"recall@5: {sum(recall_at_5s) / len(recall_at_5s)}")
    print(f"recall@10: {sum(recall_at_10s) / len(recall_at_10s)}")
    print(f"NDCG@5: {sum(ndcg_at_5s) / len(ndcg_at_5s)}")
    print(f"NDCG@10: {sum(ndcg_at_10s) / len(ndcg_at_10s)}")
    print(f"Distinct matched items count at 1: {len(dpc_1_set)}")
    print(f"Distinct matched items count at 10: {len(dpc_10_set)}")
    print(f"Total unique predicted items count at 1: {len(total_1_set)}")
    print(f"Total unique predicted items count at 10: {len(total_10_set)}")



    if eval_mode == 'Target':
        # Calculate hit metrics
        total_hit_count_at_1 = sum(hit_count_at_1_list)
        total_hit_revenue_sum_at_1 = sum(total_hit_revenue_at_1_list)
        total_hit_count_at_10 = sum(hit_count_at_10_list)
        total_hit_revenue_sum_at_10 = sum(total_hit_revenue_at_10_list)
        avg_hit_revenue_at_1 = total_hit_revenue_sum_at_1 / total_hit_count_at_1 if total_hit_count_at_1 > 0 else 0.0
        avg_hit_revenue_at_10 = total_hit_revenue_sum_at_10 / total_hit_count_at_10 if total_hit_count_at_10 > 0 else 0.0

        print(f"Hit Count @1: {total_hit_count_at_1}")
        print(f"Total Hit Revenue @1: {total_hit_revenue_sum_at_1}")
        print(f"Average Hit Revenue @1: {avg_hit_revenue_at_1}")
        print(f"Hit Count @10: {total_hit_count_at_10}")
        print(f"Total Hit Revenue @10: {total_hit_revenue_sum_at_10}")
        print(f"Average Hit Revenue @10: {avg_hit_revenue_at_10}")
        model.train()
        return (sum(recall_at_1s) / len(recall_at_1s), sum(recall_at_5s) / len(recall_at_5s),
                sum(recall_at_10s) / len(recall_at_10s), sum(ndcg_at_5s) / len(ndcg_at_5s),
                sum(ndcg_at_10s) / len(ndcg_at_10s), sum(losses) / len(losses),
                len(dpc_1_set), len(dpc_10_set), len(total_1_set), len(total_10_set), total_hit_count_at_1, total_hit_revenue_sum_at_1,
                avg_hit_revenue_at_1, total_hit_count_at_10, total_hit_revenue_sum_at_10,
                avg_hit_revenue_at_10)
    else:
        model.train()
        return (sum(recall_at_1s) / len(recall_at_1s), sum(recall_at_5s) / len(recall_at_5s),
                sum(recall_at_10s) / len(recall_at_10s), sum(ndcg_at_5s) / len(ndcg_at_5s),
                sum(ndcg_at_10s) / len(ndcg_at_10s), sum(losses) / len(losses),
                len(dpc_1_set), len(dpc_10_set), len(total_1_set), len(total_10_set),)

def evaluate_in_train(model, dataloader, device, item_len, num_beams=10, eval_mode = 'Target', no_output=False, behavior_token = True, reverse_bt = False):
    model.eval()
    recall_at_1s = []
    recall_at_5s = []
    recall_at_10s = []
    ndcg_at_5s = []
    ndcg_at_10s = []
    losses = []
    total_correct = 0
    total_samples = 0
    total_true_positives = 0
    total_predicted_positives = 0
    total_actual_positives = 0
    dpc_1, dpc_10, total_1, total_10 = set(), set(), set(), set()
    if not no_output:
        progress_bar = tqdm(range(len(dataloader)))
    for batch in dataloader:
        batch_size = batch['input_ids'].shape[0]
        input_ids = batch['input_ids'].to(device).to(torch.long)
        attention_mask = batch['attention_mask'].to(device).to(torch.long)
        labels = batch['labels'].to(device).to(torch.long)
        label_len = labels.shape[1]

        decoder_input = torch.zeros(batch_size, 1, device=input_ids.device, dtype=torch.long)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            losses.append(outputs.loss.item())
            del(outputs)
        if behavior_token and (not reverse_bt):
            outputs = beam_search(model, input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input, max_length=label_len + 1, num_beams=num_beams, num_return_sequences=10)
        else:
            outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input, max_length=label_len + 1, num_beams=num_beams, num_return_sequences=10)
        outputs = outputs[:, 1:-1].reshape(batch_size, 10, -1)  # Remove BOS and EOS
        labels = labels[:,:-1] # Remove EOS

        batch_dpc_1, batch_dpc_10, batch_total_1, batch_total_10 = dpc(labels, outputs)
        dpc_1.update(batch_dpc_1)
        dpc_10.update(batch_dpc_10)
        total_1.update(batch_total_1)
        total_10.update(batch_total_10)

        recall_at_1, recall_at_5, recall_at_10, ndcg_at_5, ndcg_at_10 = calculate_metrics(outputs, labels, eval_mode=eval_mode)
        recall_at_1s.append(recall_at_1)
        recall_at_5s.append(recall_at_5)
        recall_at_10s.append(recall_at_10)
        ndcg_at_5s.append(ndcg_at_5)
        ndcg_at_10s.append(ndcg_at_10)


        outputs_bt = outputs[:, 0, 0]
        labels_bt = labels[:, 0]
        correct, batch_size, tp, pred_pos, actual_pos = calculate_acc_recall_precision(outputs_bt, labels_bt)
        
        total_correct += correct
        total_samples += batch_size
        total_true_positives += tp
        total_predicted_positives += pred_pos
        total_actual_positives += actual_pos
        if not no_output:
            progress_bar.set_description(f"recall@10: {(sum(recall_at_10s) / len(recall_at_10s)):.4f}, NDCG@10: {(sum(ndcg_at_10s) / len(ndcg_at_10s)):.4f}")
            progress_bar.update(1)
    if not no_output:
        progress_bar.close()
    print(f"Validation Loss: {sum(losses) / len(losses)}")
    print(f"recall@1: {sum(recall_at_1s) / len(recall_at_1s)}")
    print(f"recall@5: {sum(recall_at_5s) / len(recall_at_5s)}")
    print(f"recall@10: {sum(recall_at_10s) / len(recall_at_10s)}")
    print(f"NDCG@5: {sum(ndcg_at_5s) / len(ndcg_at_5s)}")
    print(f"NDCG@10: {sum(ndcg_at_10s) / len(ndcg_at_10s)}")
    print(f"Distinct matched items count at 1: {len(dpc_1)}")
    print(f"Distinct matched items count at 10: {len(dpc_10)}")
    print(f"Total unique predicted items count at 1: {len(total_1)}")
    print(f"Total unique predicted items count at 10: {len(total_10)}")
    print("--- Behavior-only evaluation ---")
    accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    precision = total_true_positives / total_predicted_positives if total_predicted_positives > 0 else 0.0
    recall = total_true_positives / total_actual_positives if total_actual_positives > 0 else 0.0
    print(f"Total correct predictions: {total_correct}")
    print(f"Total predicted positives: {total_predicted_positives}")
    print(f"Total actual positives: {total_actual_positives}")
    print(f"Total True Positives: {total_true_positives}")
    print(f"Total number of samples: {total_samples}")
    print(f"Accuracy (Behavior_only): {accuracy}")
    print(f"Recall (Behavior_only): {recall}")
    print(f"Precision (Behavior_only): {precision}")

    model.train()
    return (
        sum(recall_at_1s) / len(recall_at_1s), 
        sum(recall_at_5s) / len(recall_at_5s), 
        sum(recall_at_10s) / len(recall_at_10s), 
        sum(ndcg_at_5s) / len(ndcg_at_5s), 
        sum(ndcg_at_10s) / len(ndcg_at_10s), 
        sum(losses) / len(losses), 
        len(dpc_1), 
        len(dpc_10), 
        len(total_1), 
        len(total_10), 
        accuracy,
        recall, 
        precision, 
        total_true_positives
    )