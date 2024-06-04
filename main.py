from InquirerPy import inquirer
from mlp_xor_pytorch import cli_run_mlp_pytorch
from mlp_xor import cli_run_mlp_na_mao

def choose_model():
    choices = [
        {"name": "Com PyTorch", "value": "pytorch"},
        {"name": "Na marra", "value": "na-marra"}
    ]
    model_type = inquirer.select(
        message="Escolha a implementação do modelo MLP:",
        choices=choices,
    ).execute()
    return model_type

def main():
    model_type = choose_model()
    if model_type == "pytorch":
        cli_run_mlp_pytorch()
    elif model_type == "na-marra":
        cli_run_mlp_na_mao()
    else:
        print("Inválido. Saindo...")
        exit(1)

if __name__ == "__main__":
    main()