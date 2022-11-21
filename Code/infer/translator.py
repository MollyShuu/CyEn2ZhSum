import torch

from infer.beam import Beam


def cross_beam_search(opt, model, Gsens, Gsens_adjs, Gcpts, probs, idxes, fields):
    batch_size = Gcpts.size(0)
    n_nodes = Gsens.size(1)
    beam_size = opt.beam_size
    device = Gcpts.device
    num_words = model.tgt_vocab_size

    graph_encoder = model.graph_encoder
    cpt_encoder = model.cpt_encoder
    decoder = model.decoder
    projection = model.projection
    sm = model.sm

    beams = [Beam(opt.beam_size, fields["tgt"].pad_id, fields["tgt"].bos_id, fields["tgt"].eos_id,
                  device, opt.length_penalty) for _ in range(batch_size)]
    Gsens = Gsens.repeat(1, 1, beam_size).view(batch_size * beam_size, n_nodes, -1)
    Gsens_adjs = Gsens_adjs.repeat(1, 1, beam_size).view(batch_size * beam_size, n_nodes, -1)
    Gcpts = Gcpts.repeat(1, beam_size).view(batch_size * beam_size, -1)
    src_len_size, n_words = probs.size(1), probs.size(2)

    probs = probs.repeat(1, beam_size, 1).view(batch_size * beam_size, src_len_size, -1)
    idxes = idxes.repeat(1, beam_size, 1).view(batch_size * beam_size, src_len_size, -1)

    enc_outputs = graph_encoder(Gsens, Gsens_adjs)
    #  print(enc_outputs.size(), enc_outputs)
    cpt_outputs, enc_cpt_attns = cpt_encoder(Gcpts)

    beam_expander = (torch.arange(batch_size) * beam_size).view(-1, 1).to(device)

    previous = None

    for i in range(opt.max_length):
        if all((b.done for b in beams)):
            break

        current_token = torch.cat([b.current_state for b in beams]).unsqueeze(-1)
        dec_outputs, p_range, previous, dec_enc_attns, weights = decoder(current_token, Gsens, enc_outputs,
                                                                         Gcpts, cpt_outputs, previous, i)

        previous_score = torch.stack([b.scores for b in beams]).unsqueeze(-1)
        out = sm(projection(dec_outputs))
        bsize, _, tgtvocab = out.size()
        _, srclen, _ = cpt_outputs.size()
        tmp_trans_scores = torch.zeros(bsize, srclen, tgtvocab).to(device)
        tmp_trans_scores.scatter_add_(2, idxes, probs)
        trans_scores = torch.matmul(weights.float(), tmp_trans_scores)
        del tmp_trans_scores
        out = torch.log(p_range * trans_scores + (1 - p_range) * out)
        out = out.view(batch_size, beam_size, -1)

        if i < opt.min_length:
            out[:, :, fields["tgt"].eos_id] = -1e15

        # find topk candidates
        scores, indexes = (out + previous_score).view(batch_size, -1).topk(beam_size)

        # find origins and token
        origins = (indexes.view(-1) // num_words).view(batch_size, beam_size)
        tokens = (indexes.view(-1) % num_words).view(batch_size, beam_size)

        for j, b in enumerate(beams):
            b.advance(scores[j], origins[j], tokens[j])

        origins = (origins + beam_expander).view(-1)
        previous = torch.index_select(previous, 0, origins)

    return [b.best_hypothesis for b in beams]



def cross_beam_g_search(opt, model, Gsens, Gsens_adjs, fields):
    batch_size = Gsens.size(0)
    n_nodes = Gsens.size(1)
    beam_size = opt.beam_size
    device = Gsens.device
    num_words = model.tgt_vocab_size

    graph_encoder = model.graph_encoder
    decoder = model.decoder
    projection = model.projection
    sm = model.sm

    beams = [Beam(opt.beam_size, fields["tgt"].pad_id, fields["tgt"].bos_id, fields["tgt"].eos_id,
                  device, opt.length_penalty) for _ in range(batch_size)]
    Gsens = Gsens.repeat(1, 1, beam_size).view(batch_size * beam_size, n_nodes, -1)
    Gsens_adjs = Gsens_adjs.repeat(1, 1, beam_size).view(batch_size * beam_size, n_nodes, -1)
    enc_outputs = graph_encoder(Gsens, Gsens_adjs)

    beam_expander = (torch.arange(batch_size) * beam_size).view(-1, 1).to(device)

    previous = None

    for i in range(opt.max_length):
        if all((b.done for b in beams)):
            break

        # [batch_size x beam_size, 1]
        current_token = torch.cat([b.current_state for b in beams]).unsqueeze(-1)
        dec_outputs, previous = decoder(current_token, Gsens, enc_outputs, previous, i)
        previous_score = torch.stack([b.scores for b in beams]).unsqueeze(-1)
        out = torch.log(sm(projection(dec_outputs)))

        out = out.view(batch_size, beam_size, -1)

        if i < opt.min_length:
            out[:, :, fields["tgt"].eos_id] = -1e15

        # find topk candidates
        scores, indexes = (out + previous_score).view(batch_size, -1).topk(beam_size)

        # find origins and token
        origins = (indexes.view(-1) // num_words).view(batch_size, beam_size)
        tokens = (indexes.view(-1) % num_words).view(batch_size, beam_size)

        for j, b in enumerate(beams):
            b.advance(scores[j], origins[j], tokens[j])

        origins = (origins + beam_expander).view(-1)
        previous = torch.index_select(previous, 0, origins)

    return [b.best_hypothesis for b in beams]




def gcsc_beam_search(opt, model, Gcpts, probs, idxes, fields):
    batch_size = Gcpts.size(0)
    beam_size = opt.beam_size
    device = Gcpts.device
    num_words = model.tgt_vocab_size

    cpt_encoder = model.cpt_encoder
    decoder = model.decoder
    projection = model.projection
    sm = model.sm

    beams = [Beam(opt.beam_size, fields["tgt"].pad_id, fields["tgt"].bos_id, fields["tgt"].eos_id,
                  device, opt.length_penalty) for _ in range(batch_size)]
    Gcpts = Gcpts.repeat(1, beam_size).view(batch_size * beam_size, -1)
    src_len_size, n_words = probs.size(1), probs.size(2)

    probs = probs.repeat(1, beam_size, 1).view(batch_size * beam_size, src_len_size, -1)
    idxes = idxes.repeat(1, beam_size, 1).view(batch_size * beam_size, src_len_size, -1)

    cpt_outputs, enc_cpt_attns = cpt_encoder(Gcpts)

    beam_expander = (torch.arange(batch_size) * beam_size).view(-1, 1).to(device)

    previous = None

    for i in range(opt.max_length):
        if all((b.done for b in beams)):
            break

        # [batch_size x beam_size, 1]
        current_token = torch.cat([b.current_state for b in beams]).unsqueeze(-1)
        dec_outputs, p_range, previous, weights = decoder(current_token, Gcpts, cpt_outputs, previous, i)
        previous_score = torch.stack([b.scores for b in beams]).unsqueeze(-1)
        out = sm(projection(dec_outputs))
        bsize, _, tgtvocab = out.size()
        _, srclen, _ = cpt_outputs.size()
        tmp_trans_scores = torch.zeros(bsize, srclen, tgtvocab).cuda()
        tmp_trans_scores.scatter_add_(2, idxes, probs)
        trans_scores = torch.matmul(weights.float(), tmp_trans_scores)
        del tmp_trans_scores
        out = torch.log(p_range * trans_scores + (1 - p_range) * out)
        out = out.view(batch_size, beam_size, -1)

        if i < opt.min_length:
            out[:, :, fields["tgt"].eos_id] = -1e15

        # find topk candidates
        scores, indexes = (out + previous_score).view(batch_size, -1).topk(beam_size)

        # find origins and token
        origins = (indexes.view(-1) // num_words).view(batch_size, beam_size)
        tokens = (indexes.view(-1) % num_words).view(batch_size, beam_size)

        for j, b in enumerate(beams):
            b.advance(scores[j], origins[j], tokens[j])

        origins = (origins + beam_expander).view(-1)
        previous = torch.index_select(previous, 0, origins)

    return [b.best_hypothesis for b in beams]
