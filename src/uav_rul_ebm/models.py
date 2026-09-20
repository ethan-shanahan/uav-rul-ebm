from torch import Tensor, nn


class Baseline(nn.Module):
    """Convolutional-recurrent baseline for remaining useful life prediction.

    The model applies a one-dimensional convolution across each input window,
    models the extracted sequence with an LSTM, and predicts one RUL value per
    window.
    """

    def __init__(
        self,
        num_features: int,
        cnn_out_channels: int = 32,
        lstm_hidden_dim: int = 64,
        dropout_p: float = 0.3,
    ) -> None:
        """Initialize the baseline model.

        Args:
            num_features: Number of features in each timestep.
            cnn_out_channels: Number of channels produced by the convolution.
            lstm_hidden_dim: Number of features in the LSTM hidden state.
            dropout_p: Dropout probability used after the convolution and in
                the regression head.
        """
        super().__init__()
        if num_features < 1:
            raise ValueError("num_features must be positive")
        self.config = {
            "num_features": num_features,
            "cnn_out_channels": cnn_out_channels,
            "lstm_hidden_dim": lstm_hidden_dim,
            "dropout_p": dropout_p,
        }

        self.conv1d = nn.Conv1d(
            in_channels=num_features,
            out_channels=cnn_out_channels,
            kernel_size=3,
            padding=1,
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_p)

        self.lstm = nn.LSTM(
            input_size=cnn_out_channels,
            hidden_size=lstm_hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        self.regression_head = nn.Sequential(
            nn.Linear(lstm_hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(32, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Predict remaining useful life for each input window.

        Args:
            x: Input tensor with shape ``(batch, window_size, num_features)``.

        Returns:
            A tensor with shape ``(batch, 1)`` containing one RUL prediction
            per input window.
        """
        x = x.permute(0, 2, 1)

        x = self.conv1d(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = x.permute(0, 2, 1)

        _, (hidden_state, _) = self.lstm(x)
        final_hidden_state = hidden_state[-1]

        return self.regression_head(final_hidden_state)
