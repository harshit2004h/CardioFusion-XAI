import torch
import torch.nn as nn


class Swish(nn.Module):
    """
    Swish activation:
        f(x) = x * sigmoid(x)
    """

    def forward(self, x):
        return x * torch.sigmoid(x)


class ConvNormPool(nn.Module):
    """
    CNN feature extraction block.

    Structure:
        Conv1D
        BatchNorm
        Swish
        Conv1D
        BatchNorm
        Swish
        MaxPool
        Dropout
    """

    def __init__(
        self,
        input_channels,
        output_channels,
        kernel_size=5,
        pool_size=2,
        dropout=0.30,
    ):
        super().__init__()

        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(
            in_channels=input_channels,
            out_channels=output_channels,
            kernel_size=kernel_size,
            padding=padding,
        )

        self.bn1 = nn.BatchNorm1d(
            output_channels
        )

        self.conv2 = nn.Conv1d(
            in_channels=output_channels,
            out_channels=output_channels,
            kernel_size=kernel_size,
            padding=padding,
        )

        self.bn2 = nn.BatchNorm1d(
            output_channels
        )

        self.activation = Swish()

        self.pool = nn.MaxPool1d(
            kernel_size=pool_size
        )

        self.dropout = nn.Dropout(
            dropout
        )

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x)

        x = self.pool(x)

        x = self.dropout(x)

        return x


class TemporalAttention(nn.Module):
    """
    Temporal attention mechanism.

    Input:
        [batch, time, features]

    Output:
        context:
            [batch, features]

        weights:
            [batch, time, 1]
    """

    def __init__(
        self,
        input_size,
    ):
        super().__init__()

        hidden_attention = max(
            input_size // 2,
            1,
        )

        self.score = nn.Sequential(

            nn.Linear(
                input_size,
                hidden_attention,
            ),

            nn.Tanh(),

            nn.Linear(
                hidden_attention,
                1,
            ),
        )

    def forward(self, x):

        scores = self.score(x)

        # Normalize attention over time.
        weights = torch.softmax(
            scores,
            dim=1,
        )

        context = torch.sum(
            weights * x,
            dim=1,
        )

        return context, weights


class CNNBiLSTMAttention(nn.Module):
    """
    PTB-XL CNN + BiLSTM + Attention model.

    Input:
        [B, 12, 1000]

    Pipeline:
        12-lead ECG
            ↓
        CNN block 1
            ↓
        CNN block 2
            ↓
        2-layer BiLSTM
            ↓
        Temporal Attention
            ↓
        Shared FC
            ↓
        Diagnostic head
        Rhythm head
    """

    def __init__(
        self,
        num_leads,
        num_diagnostic_classes,
        num_rhythm_classes,
        cnn_channels=128,
        lstm_hidden=128,
        lstm_layers=2,
        dropout=0.30,
    ):
        super().__init__()

        self.cnn1 = ConvNormPool(
            input_channels=num_leads,
            output_channels=cnn_channels,
            kernel_size=7,
            pool_size=2,
            dropout=dropout,
        )

        self.cnn2 = ConvNormPool(
            input_channels=cnn_channels,
            output_channels=cnn_channels,
            kernel_size=5,
            pool_size=2,
            dropout=dropout,
        )

        self.bilstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=(
                dropout
                if lstm_layers > 1
                else 0.0
            ),
        )

        lstm_output_size = (
            lstm_hidden * 2
        )

        self.attention = (
            TemporalAttention(
                lstm_output_size
            )
        )

        self.shared_fc = nn.Sequential(

            nn.Linear(
                lstm_output_size,
                128,
            ),

            nn.LayerNorm(128),

            nn.ReLU(),

            nn.Dropout(dropout),
        )

        self.diagnostic_head = nn.Linear(
            128,
            num_diagnostic_classes,
        )

        if num_rhythm_classes > 0:

            self.rhythm_head = nn.Linear(
                128,
                num_rhythm_classes,
            )

        else:

            self.rhythm_head = None

    def forward(self, x):

        # x:
        # [B, 12, 1000]

        x = self.cnn1(x)

        # [B, 128, 500]

        x = self.cnn2(x)

        # [B, 128, 250]

        # LSTM expects:
        # [B, time, features]

        x = x.transpose(
            1,
            2,
        )

        # [B, 250, 128]

        x, _ = self.bilstm(x)

        # [B, 250, 256]

        context, attention_weights = (
            self.attention(x)
        )

        # [B, 256]

        shared = self.shared_fc(
            context
        )

        diagnostic_logits = (
            self.diagnostic_head(
                shared
            )
        )

        if self.rhythm_head is not None:

            rhythm_logits = (
                self.rhythm_head(
                    shared
                )
            )

        else:

            rhythm_logits = shared.new_empty(
                (
                    shared.size(0),
                    0,
                )
            )

        # IMPORTANT:
        # Do NOT apply sigmoid/softmax here.
        #
        # BCEWithLogitsLoss expects raw logits.

        return {
            "diagnostic":
                diagnostic_logits,

            "rhythm":
                rhythm_logits,

            "attention":
                attention_weights,
        }