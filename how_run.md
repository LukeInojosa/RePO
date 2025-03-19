Instale o gerenciador de pacotes Homebrew:
    1. /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)" (instala o brew)
    2. Realizar os passos que manda no terminal após a instalação

Instalando pyenv e pyenv-virtualenv:
    1. sudo apt-get install -y build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm gettext libncurses5-dev tk-dev tcl-dev blt-dev libgdbm-dev git python-dev python3-dev aria2 vim libnss3-tools python3-venv liblzma-dev libpq-dev (instala dependências do pyenv)
    2. brew install pyenv pyenv-virtualenv
    3. Escreva o texto abaixo no final do documento ~/.bashrc para definir como variáveis de ambiente globais:
        #  export PYENV_ROOT="$HOME/.pyenv"
        #  export PATH="$PYENV_ROOT/bin:$PATH"
        #  eval "$(pyenv init --path)"
        #  eval "$(pyenv init -)"
        #  eval "$(pyenv virtualenv-init -)"
    4. exec $SHELL (aplica as alterações feitas no arquivo acima)

Configurando a versão do python:
    1. pyenv install 3.7.7 (instala python 3.7.7)
        0. pyenv versions (verifica se a versão foi instalada)
    2. pyenv virtualenv 3.7.7 RePO (cria ambiente virtual com python 3.7.7 e nome RePO)
    3. pyenv local RePO (coloca o ambiente local como ambiente virtual RePO)
        0. python -V (verifica a versão atual do python)
        0.pip -V (verifica se está num virtual enviroment)
    <!-- 4. pyenv local 3.7.7 (configura o repositório local com python 3.7.7)
        0. python --version (verica se a versão no repositório local realmente é a configurada) -->
Agora é possível instalar os pacotes neste ambiente:
    1. pip install --upgrade pip (atualiza o pip)
    2. pip install tensorflow==2.1.0
    3. pip install pandas==1.0.3
    4. pip install numpy==1.18.1
        0. pip list (verifica os pacotes e versões instaladas)
        0. pip show [nome do pacote] (verifica versão de um pacote específico)   
    5. pip install --upgrade protobuf==3.20.3 (downgrade protobuf) 

Desinstalando ambientes virtuais:
    1. pyenv uninstall RePO (remove ambiente virtual RePO)
    2. pyenv virtualenvs (verifica todos os ambientes virtuais criados)
    2. se o ambiente ainda estiver listado, remova a pasta manualmente:
        0. rm -rf ~/.pyenv/versions/RePO

Ativando e Desativando ambientes virtuais:
    1. pyenv activate RePO (outra forma de ativar ambiente virtual)
    2. source deactivate (desativa ambiente virtual)
        exec $SHELL (atualiza as alterações no sheell)