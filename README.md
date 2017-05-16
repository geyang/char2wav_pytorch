# char2wav_pytorch 

link to paper: https://openreview.net/pdf?id=B1VWyySKx

## TODOs

- [ ] Implement Model
    - [ ] Reader
        - [x] Encoder
            - [ ] make encoder bi-directional
        - [x] Decoder
            - [ ] Add Attention to decoder
    - [x] SampleRNN
        - [x] overall architecture
        - [x] perforated RNN module
- [ ] Unit Test

## Model Architecture

![char2wav](./figures/char2wav.png)