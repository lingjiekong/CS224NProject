import torch
from torch import nn
import torch.nn.functional as F


class BiLSTMA(nn.Module):
    inference_chunk_length = 512

    def __init__(self, input_features, recurrent_features):
        super().__init__()
        self.rnn = nn.LSTM(input_features, recurrent_features, batch_first=True, bidirectional=True)
        self.att_projection1 = nn.Linear(recurrent_features * 2, recurrent_features *2 , bias=False) 
        self.att_projection2 = nn.Linear(recurrent_features * 2, recurrent_features *2, bias=False) 
        self.att_projection_out_1 = nn.Linear(recurrent_features * 2, recurrent_features *2 , bias=False)
        self.att_projection_out_2 = nn.Linear(recurrent_features * 2, recurrent_features *2, bias=False)

    def forward(self, x):
        if self.training:
            
            ##FIRST METHOD OF ATTENTION-- MULTIPLICATIVE ATTENTION
            ##LATER TRY CAT ATTENTION 
            # hidden_states size is (batch, len, 2 * recurrent_features)
            hidden_states,(last_hidden, last_cell) = self.rnn(x)
            # hidden_states_proj1 is (batch, len, 2 * recurrent_features)
            hidden_states_proj1 = self.att_projection1(hidden_states)
            # hidden_states_proj2 is (batch, len,2 * recurrent_features)
            hidden_states_proj2 = self.att_projection2(hidden_states)
            # hide_transpose size (batch, 2 * recurrent_features,len)
            hidden_transpose = torch.transpose(hidden_states_proj2, 1,2)
            # attebtion score matrix  
            #  (b×n×m) * (b×m×p) -->  (b×n×p)
            #  (batch, len, recurrent_features) * (batch, 2 * recurrent_features,len) --> (batch, len,len)
            attention_matrix = torch.bmm( hidden_states_proj1, hidden_transpose)
            # apply softmax 
            attention_score_matrix = F.softmax(attention_matrix, -1)
            # (batch, 2 * recurrent_features,len) * (batch, len,len) -- > (batch,2 * recurrent_features,len)
            new_hidden_matrix_t = torch.bmm(hidden_transpose, torch.transpose(attention_score_matrix,1,2))
            # want (batch,len, 2 * recurrent_features) --> transpose
            new_hidden_matrix = torch.transpose(new_hidden_matrix_t,1,2)
            # residual connection: (batch,len, 2 * rsecurrent_features)  cat  (batch,len, 2 * recurrent_features)
            final_hidden_matrix = torch.cat((new_hidden_matrix,hidden_states),2)
            
            return final_hidden_matrix
            
        else:
            # evaluation mode: support for longer sequences that do not fit in memory
            batch_size, sequence_length, input_features = x.shape
            # print(x.shape)
            hidden_size = self.rnn.hidden_size
            num_directions = 2 if self.rnn.bidirectional else 1

            h = torch.zeros(num_directions, batch_size, hidden_size, device=x.device)
            c = torch.zeros(num_directions, batch_size, hidden_size, device=x.device)
            output = torch.zeros(batch_size, sequence_length, num_directions * hidden_size, device=x.device)

            # forward direction
            slices = range(0, sequence_length, self.inference_chunk_length)
            for start in slices:
                end = start + self.inference_chunk_length
                output[:, start:end, :], (h, c) = self.rnn(x[:, start:end, :], (h, c))


            # reverse direction
            if self.rnn.bidirectional:
                h.zero_()
                c.zero_()

                for start in reversed(slices):
                    end = start + self.inference_chunk_length
                    result, (h, c) = self.rnn(x[:, start:end, :], (h, c))
                    output[:, start:end, hidden_size:] = result[:, :, hidden_size:]

            out_proj1 = self.att_projection_out_1(output)
            out_proj2 = self.att_projection_out_2(output)
            out_transpose = torch.transpose(out_proj2, 1,2)
            out_attention_matrix = torch.bmm(out_proj1,out_transpose)
            out_attention_score_matrix = F.softmax(out_attention_matrix,dim = -1)
            out_hidden_matrix_t = torch.bmm(out_transpose, torch.transpose(out_attention_score_matrix,1,2))
            out_hidden_matrix = torch.transpose(out_hidden_matrix_t,1,2)
            final_out_matrix = torch.cat((out_hidden_matrix,output),2)
            return final_out_matrix

            # return output
