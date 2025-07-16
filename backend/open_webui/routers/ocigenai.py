import time
import asyncio
import random
import os # Importar 'os' para acessar variáveis de ambiente

from typing import List, Optional, Literal, AsyncGenerator

from pydantic import BaseModel, Field

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn

# Importar o SDK da OCI
import oci
import oci.generative_ai_inference.models as ga_models
import oci.generative_ai_inference as ga_inference
# Removida a importação de GenerativeAiInferenceClientAsync, conforme solicitado e para evitar ModuleNotFoundError.
# A implementação utiliza o cliente síncrono e asyncio.to_thread para compatibilidade.


# --- 1. Modelos Pydantic para a API de Chat (inspirados na OpenAI) ---

class Message(BaseModel):
    """Representa uma mensagem no histórico de chat."""
    role: Literal["user", "assistant", "system"]
    content: str

class ChatCompletionRequest(BaseModel):
    """Requisição para completar o chat, com opção de streaming."""
    model: str
    messages: List[Message]
    stream: Optional[bool] = False # Indica se o cliente deseja resposta em streaming

class Usage(BaseModel):
    """Estatísticas de uso de tokens."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class Choice(BaseModel):
    """Uma escolha de resposta completa do modelo."""
    index: int
    message: Message
    finish_reason: Optional[Literal["stop", "length", "content_filter", "tool_calls", "function_call"]] = None

class ChatCompletionResponse(BaseModel):
    """Resposta completa do chat (não-streaming)."""
    id: str = Field(default_factory=lambda: f"chatcmpl-{int(time.time())}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[Choice]
    usage: Optional[Usage] = None


# --- 2. Modelos Pydantic para Chunks de Streaming (inspirados na OpenAI) ---

class DeltaMessage(BaseModel):
    """Um pedaço de mensagem em um chunk de streaming."""
    role: Optional[Literal["user", "assistant", "system"]] = None
    content: Optional[str] = None
    finish_reason: Optional[Literal["stop", "length", "content_filter", "tool_calls", "function_call"]] = None

class ChatCompletionChunkChoice(BaseModel):
    """Uma escolha de resposta em um chunk de streaming."""
    index: int
    delta: DeltaMessage
    finish_reason: Optional[Literal["stop", "length", "content_filter", "tool_calls", "function_call"]] = None

class ChatCompletionChunk(BaseModel):
    """Um chunk de resposta de streaming."""
    id: str = Field(default_factory=lambda: f"chatcmpl-{int(time.time())}")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionChunkChoice]

# --- 3. Modelos Pydantic para Listagem de Modelos ---
class ModelCard(BaseModel):
    """Representa um modelo de IA disponível."""
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "OCI-GenerativeAI" 

class ModelList(BaseModel):
    """Representa a lista de modelos disponíveis."""
    object: str = "list"
    data: List[ModelCard]


# --- 4. Integração Real com o Serviço OCI Generative AI (Implementação SÍNCRONA) ---

class OCIService:
    """
    Esta classe encapsula a integração real com o OCI Generative AI Service,
    utilizando a implementação síncrona fornecida por você e lendo configurações de ambiente.
    """
    def __init__(self):
        self.model_id_oci_full = "ocid1.generativeaimodel.oc1.sa-saopaulo-1.amaaaaaask7dceyaxu7lvx6k45r2hapxtuc2q5rleaujcowq6xbcywwtzhsq"
        self.model_id_alias = "SKY OCI" # Apelido do modelo para exposição na API

        # Lendo o Compartment ID da variável de ambiente OCI_COMPARTMENT_ID
        self.compartment_id = os.environ.get(
            "OCI_COMPARTMENT_ID", 
            "ocid1.tenancy.oc1..aaaaaaaan4paxie47spf5t2iifynmadqapbspy5w7av7veuuv2moqpvii4iq" # <-- ATUALIZE AQUI COM SEU OCID REAL para fallback
        )
        
        CONFIG_PROFILE = "DEFAULT" 
        
        # Verificar se as variáveis de ambiente estão definidas, caso contrário usar valores padrão
        config_file_path = os.environ.get("OCI_CONFIG_FILE_PATH_IN_CONTAINER", "/app/.oci/config")
        key_file_path = os.environ.get("OCI_KEY_FILE_PATH_IN_CONTAINER", "/app/.oci/private.pem")
        
        print(f"Tentando carregar configuração OCI de: {config_file_path}")
        print(f"Tentando carregar chave privada de: {key_file_path}")
        
        # Verificar se os arquivos existem
        if not os.path.exists(config_file_path):
            print(f"AVISO: Arquivo de configuração não encontrado em {config_file_path}")
            # Tentar caminhos alternativos
            #alt_paths = ["/home/ricardorise/.oci/config", "/app/.oci/config", "~/.oci/config"]
            alt_paths = ["/home/ricardorise/.oci/config"]
            for path in alt_paths:
                expanded_path = os.path.expanduser(path)
                if os.path.exists(expanded_path):
                    print(f"Encontrado arquivo de configuração alternativo em: {expanded_path}")
                    config_file_path = expanded_path
                    break
        
        if not os.path.exists(key_file_path):
            print(f"AVISO: Arquivo de chave privada não encontrado em {key_file_path}")
            # Tentar caminhos alternativos
            alt_paths = ["/home/ricardorise/.oci/private.pem", "/app/.oci/private.pem", "~/.oci/private.pem"]
            for path in alt_paths:
                expanded_path = os.path.expanduser(path)
                if os.path.exists(expanded_path):
                    print(f"Encontrado arquivo de chave privada alternativo em: {expanded_path}")
                    key_file_path = expanded_path
                    break

        try:
            # Informa explicitamente ao SDK da OCI onde encontrar o arquivo de configuração
            print(f"Carregando configuração de: {config_file_path}")
            self.config = oci.config.from_file(config_file_path, CONFIG_PROFILE)
            
            # Sobrescreve o caminho da chave privada na configuração carregada
            print(f"Definindo caminho da chave privada para: {key_file_path}")
            self.config["key_file"] = key_file_path
            
        except Exception as e:
            # ## CORREÇÃO ##
            # Captura a exceção e levanta um novo erro com uma mensagem formatada corretamente.
            # As aspas internas na f-string foram trocadas para simples (') para evitar SyntaxError.
            raise RuntimeError(
                f"Erro ao carregar a configuração da OCI. Certifique-se de que o arquivo "
                f"'{config_file_path}' está presente no container e que o caminho da chave privada "
                f"'{key_file_path}' está correto. Detalhes: {e}"
            )

        self.endpoint = "https://inference.generativeai.sa-saopaulo-1.oci.oraclecloud.com"

        self.generative_ai_inference_client = oci.generative_ai_inference.GenerativeAiInferenceClient(
            config=self.config, 
            service_endpoint=self.endpoint, 
            retry_strategy=oci.retry.NoneRetryStrategy(), 
            timeout=(10, 240)
        )
        print("OCI Generative AI Inference Client (SÍNCRONO) inicializado com sucesso.")
        print(f"Usando Compartment ID: {self.compartment_id}") 
        print(f"Lendo OCI Config de: {os.environ.get('OCI_CONFIG_FILE_PATH_IN_CONTAINER')}")
        print(f"Usando Chave OCI de: {os.environ.get('OCI_KEY_FILE_PATH_IN_CONTAINER')}")


    def format_mermaid_diagrams(self, text: str) -> str:
        """
        Formata diagramas Mermaid no texto para exibição adequada.
        Procura por blocos de código Mermaid e os formata corretamente.
        """
        import re
        
        # Lista de todos os tipos de diagramas Mermaid conhecidos
        mermaid_types = [
            'graph [TBLR]D', 'flowchart [TBLR]D', 'sequenceDiagram', 'classDiagram',
            'erDiagram', 'gantt', 'pie', 'stateDiagram-v2', 'stateDiagram', 'journey',
            'gitGraph', 'requirementDiagram', 'C4Context', 'C4Container', 'C4Component',
            'C4Dynamic', 'C4Deployment', 'mindmap', 'timeline', 'sankey-beta'
        ]
        
        # Criar padrão regex para todos os tipos de diagramas
        types_pattern = '|'.join(mermaid_types).replace('[TBLR]D', '[TBLR]D')
        
        result = text
        
        # CASO ESPECIAL: Detectar diagramas Mermaid dentro de blocos HTML
        # Procurar por <div class="mermaid"> ou class='mermaid'
        html_mermaid_patterns = [
            r'<div\s+class=["\']mermaid["\']>(.*?)</div>',
            r'<div\s+class=["\']mermaid-container["\']>(.*?)</div>'
        ]
        
        for pattern in html_mermaid_patterns:
            matches = list(re.finditer(pattern, result, re.DOTALL))
            replacements = []
            
            for match in matches:
                full_match = match.group(0)
                mermaid_content = match.group(1).strip()
                
                # Verificar se o conteúdo já está formatado como código Mermaid
                if not mermaid_content.startswith('```mermaid'):
                    # Extrair apenas o código Mermaid, removendo tags HTML internas
                    inner_div_match = re.search(r'<div\s+class=["\']mermaid["\']>(.*?)</div>', mermaid_content, re.DOTALL)
                    if inner_div_match:
                        mermaid_code = inner_div_match.group(1).strip()
                    else:
                        mermaid_code = mermaid_content
                    
                    # Limpar tags HTML residuais
                    mermaid_code = re.sub(r'<[^>]+>', '', mermaid_code)
                    
                    # Formatar como bloco de código Mermaid
                    formatted_content = f"```mermaid\n{mermaid_code}\n```"
                    replacements.append((match.start(), match.end(), formatted_content))
            
            # Aplicar substituições de trás para frente
            for start, end, replacement in sorted(replacements, reverse=True):
                result = result[:start] + replacement + result[end:]
        
        # CASO 1: Detectar diagramas Mermaid sem formatação
        # Padrões para encontrar blocos de código Mermaid sem a formatação correta
        patterns = [
            # Blocos que começam com tipos de diagramas Mermaid sem formatação
            f'(?<!```mermaid\\n)(?<!```\\n)({types_pattern})\\n',
            
            # Blocos que já estão em ``` mas sem especificar mermaid
            f'```\\n({types_pattern})'
        ]
        
        for pattern in patterns:
            if pattern.startswith('(?<!```'):
                # Encontrar todas as ocorrências do padrão
                matches = list(re.finditer(pattern, result))
                
                # Lista para armazenar os índices e substituições
                replacements = []
                
                for match in matches:
                    start = match.start()
                    # Verificar se não está dentro de um bloco de código existente
                    code_block_before = re.search(r'```.*?$', result[max(0, start-100):start], re.MULTILINE)
                    code_block_after = re.search(r'^.*?```', result[start:min(start+100, len(result))], re.MULTILINE)
                    
                    # Se não estiver dentro de um bloco de código existente
                    if not (code_block_before and not code_block_after):
                        replacements.append((start, f"```mermaid\n{match.group(0)}"))
                
                # Aplicar substituições de trás para frente para não afetar os índices
                for start, replacement in sorted(replacements, reverse=True):
                    result = result[:start] + replacement + result[start + len(match.group(0)):]
            
            # Caso 2: Substituir ``` por ```mermaid
            elif pattern.startswith('```\\n'):
                result = re.sub(pattern, '```mermaid\n\\1', result)
        
        # CASO 3: Garantir que blocos de código Mermaid sejam fechados corretamente
        # Encontrar todos os blocos que começam com ```mermaid
        mermaid_starts = list(re.finditer(r'```mermaid\n', result))
        
        # Para cada início de bloco mermaid
        replacements = []
        for i, start_match in enumerate(mermaid_starts):
            start_pos = start_match.end()
            
            # Determinar o fim do bloco
            if i < len(mermaid_starts) - 1:
                # Se há outro bloco mermaid depois, procurar ``` antes do próximo bloco
                next_start = mermaid_starts[i+1].start()
                end_match = re.search(r'```', result[start_pos:next_start])
                if not end_match:
                    # Não encontrou fechamento, adicionar antes do próximo bloco
                    replacements.append((next_start, '\n```\n'))
            else:
                # Último ou único bloco, procurar até o final do texto
                end_match = re.search(r'```', result[start_pos:])
                if not end_match:
                    # Não encontrou fechamento, adicionar ao final
                    replacements.append((len(result), '\n```'))
        
        # Aplicar substituições de trás para frente
        for pos, replacement in sorted(replacements, reverse=True):
            result = result[:pos] + replacement + result[pos:]
        
        # CASO 4: Detectar diagramas Mermaid em exemplos de código HTML
        # Procurar por exemplos de código HTML que contenham diagramas Mermaid
        html_code_pattern = r'```html.*?<div\s+class=["\']mermaid["\']>(.*?)</div>.*?```'
        matches = list(re.finditer(html_code_pattern, result, re.DOTALL))
        
        for match in matches:
            full_match = match.group(0)
            mermaid_content = match.group(1).strip()
            
            # Adicionar um bloco Mermaid formatado após o bloco HTML
            formatted_block = f"{full_match}\n\n```mermaid\n{mermaid_content}\n```"
            result = result.replace(full_match, formatted_block)
        
        # Verificação final: garantir que não há blocos mermaid aninhados
        result = re.sub(r'```mermaid\n(.*?)```mermaid\n', r'```mermaid\n\1', result, flags=re.DOTALL)
        
        # Garantir que não há blocos vazios
        result = re.sub(r'```mermaid\n\s*```', '', result)
        
        # Adicionar uma linha em branco após o fechamento dos blocos para melhor formatação
        result = re.sub(r'```(\n(?!\n))', '```\n\n', result)
        
        return result

    def markdown_to_html(self, text: str) -> str:
        """
        Anteriormente convertia texto Markdown para HTML.
        Agora retorna o texto Markdown diretamente, pois o OpenWebUI espera Markdown.
        """
        # Como o OpenWebUI espera Markdown e não HTML, vamos apenas retornar o texto formatado
        return text
        
    def ensure_markdown_formatting(self, text: str) -> str:
        """
        Garante que a formatação Markdown seja preservada corretamente.
        Verifica e corrige problemas comuns de formatação.
        """
        import re
        
        result = text
        
        # PARTE 1: PRESERVAR ELEMENTOS HTML
        
        # Preservar listas HTML ordenadas
        # Exemplo: "<ol><li>Item 1</li><li>Item 2</li></ol>" -> manter formatado
        ol_pattern = r'<ol>(.*?)</ol>'
        ol_matches = list(re.finditer(ol_pattern, result, re.DOTALL))
        for match in ol_matches:
            content = match.group(1)
            # Garantir que cada <li> esteja em uma nova linha
            formatted_content = re.sub(r'<li>', '\n<li>', content)
            # Substituir no texto original
            result = result.replace(match.group(0), f"<ol>{formatted_content}</ol>")
        
        # Preservar listas HTML não ordenadas
        # Exemplo: "<ul><li>Item 1</li><li>Item 2</li></ul>" -> manter formatado
        ul_pattern = r'<ul>(.*?)</ul>'
        ul_matches = list(re.finditer(ul_pattern, result, re.DOTALL))
        for match in ul_matches:
            content = match.group(1)
            # Garantir que cada <li> esteja em uma nova linha
            formatted_content = re.sub(r'<li>', '\n<li>', content)
            # Substituir no texto original
            result = result.replace(match.group(0), f"<ul>{formatted_content}</ul>")
        
        # Preservar parágrafos HTML
        # Exemplo: "<p>Parágrafo 1</p><p>Parágrafo 2</p>" -> manter formatado
        result = re.sub(r'</p>\s*<p>', '</p>\n\n<p>', result)
        
        # PARTE 2: FORMATAÇÃO MARKDOWN
        
        # Preservar quebras de linha existentes
        # Substituir quebras de linha simples por quebras de linha duplas para garantir nova linha no Markdown
        result = re.sub(r'(?<!\n)\n(?!\n)', '\n\n', result)
        
        # Garantir que listas numeradas tenham quebras de linha
        # Exemplo: "1. Item 1 2. Item 2" -> "1. Item 1\n\n2. Item 2"
        result = re.sub(r'(\d+\. .+?)(?=\s\d+\.\s|$)', r'\1\n\n', result)
        
        # Garantir que listas com marcadores tenham quebras de linha
        # Exemplo: "* Item 1 * Item 2" -> "* Item 1\n\n* Item 2"
        result = re.sub(r'(\* .+?)(?=\s\*\s|$)', r'\1\n\n', result)
        result = re.sub(r'(- .+?)(?=\s-\s|$)', r'\1\n\n', result)
        
        # Garantir que tabelas tenham quebras de linha
        # Exemplo: "| A | B | | --- | --- |" -> "| A | B |\n| --- | --- |"
        result = re.sub(r'(\|.+?\|)(?=\s\|)', r'\1\n', result)
        
        # Garantir que blocos de código tenham quebras de linha
        # Exemplo: "```python print('hello')```" -> "```python\nprint('hello')\n```"
        result = re.sub(r'```(\w*)\s+(.+?)```', r'```\1\n\2\n```', result, flags=re.DOTALL)
        
        # Detectar e formatar blocos de código que não estão corretamente marcados
        # Procurar por padrões como "python # arquivo: nome.py def função():" e formatá-los corretamente
        code_pattern = r'(?:^|\n)(?:python|javascript|java|cpp|csharp|ruby|go|php|swift|kotlin|rust|typescript)\s+(?:#|//|/\*|\*)\s+(?:arquivo|file):\s+[\w\d_.-]+\.\w+\s+(.+?)(?=\n\n|$)'
        code_matches = list(re.finditer(code_pattern, result, re.DOTALL | re.IGNORECASE))
        for match in code_matches:
            full_match = match.group(0)
            language = re.match(r'(?:python|javascript|java|cpp|csharp|ruby|go|php|swift|kotlin|rust|typescript)', full_match).group(0)
            code_content = match.group(1).strip()
            # Substituir no texto original
            result = result.replace(full_match, f"```{language}\n{code_content}\n```")
        
        # Detectar e formatar blocos de código que estão dentro de tags <code>
        code_tag_pattern = r'<code>\s*(.+?)\s*</code>'
        code_tag_matches = list(re.finditer(code_tag_pattern, result, re.DOTALL))
        for match in code_tag_matches:
            code_content = match.group(1).strip()
            # Tentar identificar a linguagem pelo conteúdo
            language = "" 
            if re.search(r'def\s+\w+\s*\(|import\s+\w+|from\s+\w+\s+import', code_content):
                language = "python"
            elif re.search(r'function\s+\w+\s*\(|var\s+\w+\s*=|const\s+\w+\s*=|let\s+\w+\s*=', code_content):
                language = "javascript"
            # Substituir no texto original
            result = result.replace(match.group(0), f"```{language}\n{code_content}\n```")
        
        # Garantir que citações tenham quebras de linha
        # Exemplo: "> Linha 1 > Linha 2" -> "> Linha 1\n\n> Linha 2"
        result = re.sub(r'(>.+?)(?=\s>|$)', r'\1\n\n', result)
        
        # Garantir que parágrafos tenham quebras de linha dupla
        # Exemplo: "Parágrafo 1 Parágrafo 2" -> "Parágrafo 1\n\nParágrafo 2"
        result = re.sub(r'(\. )([A-Z][^\n])', r'\1\n\n\2', result)
        
        # Garantir que h1, h2, h3, etc. tenham quebras de linha antes e depois
        # Exemplo: "Texto # Título" -> "Texto\n\n# Título"
        result = re.sub(r'([^\n])\s*(#+\s+[^\n]+)', r'\1\n\n\2', result)
        # Exemplo: "# Título Texto" -> "# Título\n\nTexto"
        result = re.sub(r'(#+\s+[^\n]+)\s+([^\n])', r'\1\n\n\2', result)
        
        # Garantir que cada item de lista tenha espaço após o marcador
        # Exemplo: "*Item" -> "* Item"
        result = re.sub(r'(^|\n)(\*|-)(?!\s)', r'\1\2 ', result)
        
        # Garantir que blocos de código Mermaid estejam bem formatados
        # Exemplo: ```mermaid graph TD; A-->B;``` -> ```mermaid\ngraph TD;\nA-->B;\n```
        mermaid_pattern = r'```mermaid\s+(.+?)```'
        mermaid_matches = list(re.finditer(mermaid_pattern, result, re.DOTALL))
        for match in mermaid_matches:
            content = match.group(1).strip()
            # Substituir ponto e vírgula por quebras de linha
            formatted_content = re.sub(r';\s*', ';\n', content)
            # Substituir no texto original
            result = result.replace(match.group(0), f"```mermaid\n{formatted_content}\n```")
        
        # Adicionar espaço após dois pontos em definições
        result = re.sub(r'(\w):(?!\s|\n|/|\\)', r'\1: ', result)
        
        # Normalizar múltiplas quebras de linha para no máximo duas
        result = re.sub(r'\n{3,}', '\n\n', result)
        
        # Garantir que blocos de código com linguagem especificada estejam formatados corretamente
        # Exemplo: ```python\n\nprint('hello')\n\n``` -> ```python\nprint('hello')\n```
        result = re.sub(r'```(\w+)\n\n', r'```\1\n', result)
        result = re.sub(r'\n\n```', r'\n```', result)
        
        return result

    def chat_sync(self, payload: str) -> str:
        """
        Faz uma chamada de chat SÍNCRONA para a OCI Generative AI,
        usando a sua implementação fornecida.
        """
        print(f"Chamando OCI GenAI (síncrono) com payload: '{payload}'")

        chat_detail = oci.generative_ai_inference.models.ChatDetails()

        chat_request = oci.generative_ai_inference.models.CohereChatRequest()
        chat_request.message = payload
        chat_request.max_tokens = 600
        chat_request.temperature = 1
        chat_request.frequency_penalty = 0
        chat_request.top_p = 0.75
        chat_request.top_k = 0

        chat_detail.serving_mode = oci.generative_ai_inference.models.OnDemandServingMode(model_id=self.model_id_oci_full)
        chat_detail.chat_request = chat_request
        chat_detail.compartment_id = self.compartment_id 

        try:
            chat_response = self.generative_ai_inference_client.chat(chat_detail)
            
            if chat_response and chat_response.data and chat_response.data.chat_response and chat_response.data.chat_response.text:
                reply = chat_response.data.chat_response.text
                # Formatar diagramas Mermaid na resposta
                reply = self.format_mermaid_diagrams(reply)
                # Garantir que a formatação Markdown seja preservada
                reply = self.ensure_markdown_formatting(reply)
                # Converter Markdown para HTML para melhor renderização no frontend
                reply = self.markdown_to_html(reply)
            else:
                print(f"Aviso: 'chat_response.data.chat_response.text' não encontrado. "
                      f"Verifique a estrutura da resposta completa: {chat_response.data}")
                reply = str(chat_response.data) 

            print(f"Resposta OCI (síncrona) recebida (completa, primeiros 100 caracteres): {reply[:100]}...")
            return reply
        except oci.exceptions.ServiceError as e:
            print(f"Erro do serviço OCI ao chamar chat (síncrono): HTTP Status={e.status}, Código={e.code}, Mensagem={e.message}")
            raise HTTPException(status_code=e.status, detail=f"Erro da OCI Generative AI: {e.message}")
        except Exception as e:
            print(f"Erro inesperado ao chamar OCI GenAI (síncrono): {type(e).__name__}: {e}")
            raise HTTPException(status_code=500, detail=f"Erro interno ao comunicar com OCI GenAI: {str(e)}")
            
    async def chat_async(self, payload: str) -> str:
        """
        Faz uma chamada de chat ASSÍNCRONA para a OCI Generative AI,
        usando asyncio.to_thread para não bloquear o event loop.
        """
        print(f"Chamando OCI GenAI (assíncrono) com payload: '{payload}'")

        # Preparando os detalhes do chat da mesma forma que na versão síncrona
        chat_detail = oci.generative_ai_inference.models.ChatDetails()

        chat_request = oci.generative_ai_inference.models.CohereChatRequest()
        chat_request.message = payload
        chat_request.max_tokens = 600
        chat_request.temperature = 1
        chat_request.frequency_penalty = 0
        chat_request.top_p = 0.75
        chat_request.top_k = 0

        chat_detail.serving_mode = oci.generative_ai_inference.models.OnDemandServingMode(model_id=self.model_id_oci_full)
        chat_detail.chat_request = chat_request
        chat_detail.compartment_id = self.compartment_id 

        try:
            # Executando a chamada síncrona em uma thread separada para não bloquear o event loop
            chat_response = await asyncio.to_thread(
                self.generative_ai_inference_client.chat, 
                chat_detail
            )
            
            if chat_response and chat_response.data and chat_response.data.chat_response and chat_response.data.chat_response.text:
                reply = chat_response.data.chat_response.text
                # Formatar diagramas Mermaid na resposta
                reply = self.format_mermaid_diagrams(reply)
                # Garantir que a formatação Markdown seja preservada
                reply = self.ensure_markdown_formatting(reply)
                # Converter Markdown para HTML para melhor renderização no frontend
                reply = self.markdown_to_html(reply)
            else:
                print(f"Aviso: 'chat_response.data.chat_response.text' não encontrado. "
                      f"Verifique a estrutura da resposta completa: {chat_response.data}")
                reply = str(chat_response.data) 

            print(f"Resposta OCI (assíncrona) recebida (completa, primeiros 100 caracteres): {reply[:100]}...")
            return reply
        except oci.exceptions.ServiceError as e:
            print(f"Erro do serviço OCI ao chamar chat (assíncrono): HTTP Status={e.status}, Código={e.code}, Mensagem={e.message}")
            raise HTTPException(status_code=e.status, detail=f"Erro da OCI Generative AI: {e.message}")
        except Exception as e:
            print(f"Erro inesperado ao chamar OCI GenAI (assíncrono): {type(e).__name__}: {e}")
            raise HTTPException(status_code=500, detail=f"Erro interno ao comunicar com OCI GenAI: {str(e)}")

    async def generate_streaming_response(self, user_message: str) -> AsyncGenerator[DeltaMessage, None]:
        """
        Chama a OCI GenAI de forma assíncrona usando o método chat_async, 
        obtém a resposta COMPLETA, e então simula o streaming dividindo-a em partes 
        para o cliente FastAPI.
        """
        full_response = await self.chat_async(user_message) 
        
        yield DeltaMessage(role="assistant")

        words = full_response.split()
        for i, word in enumerate(words):
            yield DeltaMessage(content=word + (" " if i < len(words) - 1 else "")) 
            await asyncio.sleep(random.uniform(0.02, 0.1)) 

        yield DeltaMessage(finish_reason="stop")


# --- 5. Classe Chatbot Adaptador ---

class Chatbot:
    """
    Esta classe atua como um adaptador entre a API do seu FastAPI
    e o serviço OCI Generative AI.
    """
    def __init__(self):
        self.oci_service = OCIService() 

    async def execute_streaming(self, req: ChatCompletionRequest) -> AsyncGenerator[str, None]:
        """
        Processa uma requisição de chat no modo streaming.
        """
        user_message = next((m.content for m in reversed(req.messages) if m.role == "user"), "Olá")
        
        oci_response_generator = self.oci_service.generate_streaming_response(user_message)
        
        session_id = f"oci-chat-stream-{int(time.time())}-{random.randint(1000, 9999)}"
        
        async for delta_message in oci_response_generator:
            chunk = ChatCompletionChunk(
                id=session_id, 
                object="chat.completion.chunk",
                created=int(time.time()),
                model=self.oci_service.model_id_alias, 
                choices=[
                    ChatCompletionChunkChoice(
                        index=0,
                        delta=DeltaMessage(
                            role=delta_message.role,
                            content=delta_message.content
                        ),
                        finish_reason=delta_message.finish_reason
                    )
                ]
            )
            
            yield f"data: {chunk.model_dump_json()}\n\n"
            
            if delta_message.finish_reason:
                break
        
    async def execute(self, req: ChatCompletionRequest) -> ChatCompletionResponse:
        """
        Processa uma requisição de chat no modo não-streaming (resposta completa).
        """
        user_message = next((m.content for m in reversed(req.messages) if m.role == "user"), "Olá")
        
        # Usando o novo método assíncrono diretamente
        reply = await self.oci_service.chat_async(user_message) 
        
        prompt_tokens = len(user_message.split())
        completion_tokens = len(reply.split())
        total_tokens = prompt_tokens + completion_tokens
        
        return ChatCompletionResponse(
            id="oci-chat-response",
            object="chat.completion",
            created=int(time.time()),
            model=self.oci_service.model_id_alias, 
            choices=[
                Choice(
                    index=0,
                    message=Message(
                        role="assistant",
                        content=reply
                    ),
                    finish_reason="stop" 
                )
            ],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens
            )
        )

# --- 6. Aplicação FastAPI Principal ---
'''
app = FastAPI(
    title="OCI Chatbot Adapter API",
    description="API para integrar o Open Web UI com a OCI Generative AI, suportando streaming e listagem de modelos.",
    version="1.0.0" 
)
'''
chatbot_instance = Chatbot()

# --- Endpoint de Saúde (Health Check) ---
#@app.get("/")
async def read_root():
    """Verifica se a API está online e respondendo."""
    return {"status": "OCI Chatbot Adapter API is running!"}

# --- Endpoint: Listagem de Modelos (/v1/models) ---
#@app.get("/v1/models", response_model=ModelList)
async def list_models2():
    """
    Retorna a lista de modelos de IA disponíveis para seleção pelo cliente.
    """
    models_data = [
        ModelCard(
            id=chatbot_instance.oci_service.model_id_alias, 
            owned_by="OCI-Cohere-Command" 
        ),
    ]
    return ModelList(data=models_data)


# --- Novos Endpoints: Adicionando os métodos solicitados pelo Open Web UI ---

#@app.get("/v1/api/tags")
async def get_api_tags():
    """
    Endpoint para retornar tags ou categorias de modelos.
    Retorna uma lista vazia ou tags genéricas.
    """
    print("GET /v1/api/tags - Responding to Open Web UI request.")
    return []

#@app.get("/v1/api/ps")
async def get_api_ps():
    """
    Endpoint para retornar informações sobre 'provisioned throughput' ou status de processos.
    Retorna um placeholder.
    """
    print("GET /v1/api/ps - Responding to Open Web UI request.")
    return {
        "status": "OK",
        "message": "Informações de capacidade provisionada (PS) não implementadas ou aplicáveis neste contexto.",
        "details": []
    }

#@app.get("/v1/api/version")
async def get_api_version():
    """
    Endpoint para retornar a versão da API.
    """
    print("GET /v1/api/version - Responding to Open Web UI request.")
    return {"version": app.version}

# --- Endpoint Principal de Chat (/v1/chat/completions) ---
#@app.post("/v1/chat/completions2") 

async def chat_completation_3(req: ChatCompletionRequest):
    """
    Versão simplificada do endpoint de chat para testes.
    """
    print("Executando chat_completation_3 com:", req)

    # Verificar se o método execute é assíncrono
    if asyncio.iscoroutinefunction(chatbot_instance.execute):
        response = await chatbot_instance.execute(req)
    else:
        # Se não for assíncrono, executar em thread separada
        response = await asyncio.to_thread(chatbot_instance.execute, req)
    
    print("Resposta recebida:", response)
    return JSONResponse(content=response.model_dump())

async def chat_completions2(req: ChatCompletionRequest):
    """
    Endpoint para interagir com o chatbot.
    """

    print(req)

    try:
        if req.model != chatbot_instance.oci_service.model_id_alias:
            raise HTTPException(
                status_code=400, 
                detail=f"Modelo '{req.model}' não encontrado. Por favor, use '{chatbot_instance.oci_service.model_id_alias}'."
            )
        
        if req.stream:
            return StreamingResponse(
                chatbot_instance.execute_streaming(req), 
                media_type="text/event-stream" 
            )
        else:
            response = await chatbot_instance.execute(req) 
            return JSONResponse(content=response.model_dump())
            
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Erro inesperado no endpoint /v1/chat/completions: {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail=f"Erro interno do servidor: {str(e)}")


import asyncio
import hashlib
import json
import logging
from pathlib import Path
from typing import Literal, Optional, overload

import aiohttp
from aiocache import cached
import requests


from fastapi import Depends, FastAPI, HTTPException, Request, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from starlette.background import BackgroundTask

from open_webui.models.models import Models
from open_webui.config import (
    CACHE_DIR,
)
from open_webui.env import (
    AIOHTTP_CLIENT_SESSION_SSL,
    AIOHTTP_CLIENT_TIMEOUT,
    AIOHTTP_CLIENT_TIMEOUT_MODEL_LIST,
    ENABLE_FORWARD_USER_INFO_HEADERS,
    BYPASS_MODEL_ACCESS_CONTROL,
)
from open_webui.models.users import UserModel

from open_webui.constants import ERROR_MESSAGES
from open_webui.env import ENV, SRC_LOG_LEVELS


from open_webui.utils.payload import (
    apply_model_params_to_body_openai,
    apply_model_system_prompt_to_body,
)
from open_webui.utils.misc import (
    convert_logit_bias_input_to_json,
)

from open_webui.utils.auth import get_admin_user, get_verified_user
from open_webui.utils.access_control import has_access

#from ocigenai.ocigenai import Chatbot
#import ocigenai.ocigenai as oci


log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["OPENAI"])


##########################################
#
# Utility functions
#
##########################################


async def send_get_request(url, key=None, user: UserModel = None):
    timeout = aiohttp.ClientTimeout(total=AIOHTTP_CLIENT_TIMEOUT_MODEL_LIST)
    try:
        async with aiohttp.ClientSession(timeout=timeout, trust_env=True) as session:
            async with session.get(
                url,
                headers={
                    **({"Authorization": f"Bearer {key}"} if key else {}),
                    **(
                        {
                            "X-OpenWebUI-User-Name": user.name,
                            "X-OpenWebUI-User-Id": user.id,
                            "X-OpenWebUI-User-Email": user.email,
                            "X-OpenWebUI-User-Role": user.role,
                        }
                        if ENABLE_FORWARD_USER_INFO_HEADERS and user
                        else {}
                    ),
                },
                ssl=AIOHTTP_CLIENT_SESSION_SSL,
            ) as response:
                return await response.json()
    except Exception as e:
        # Handle connection error here
        log.error(f"Connection error: {e}")
        return None


async def cleanup_response(
    response: Optional[aiohttp.ClientResponse],
    session: Optional[aiohttp.ClientSession],
):
    if response:
        response.close()
    if session:
        await session.close()


def openai_o_series_handler(payload):
    """
    Handle "o" series specific parameters
    """
    if "max_tokens" in payload:
        # Convert "max_tokens" to "max_completion_tokens" for all o-series models
        payload["max_completion_tokens"] = payload["max_tokens"]
        del payload["max_tokens"]

    # Handle system role conversion based on model type
    if payload["messages"][0]["role"] == "system":
        model_lower = payload["model"].lower()
        # Legacy models use "user" role instead of "system"
        if model_lower.startswith("o1-mini") or model_lower.startswith("o1-preview"):
            payload["messages"][0]["role"] = "user"
        else:
            payload["messages"][0]["role"] = "developer"

    return payload


##########################################
#
# API routes
#
##########################################

router = APIRouter()


@router.get("/config")
async def get_config(request: Request, user=Depends(get_admin_user)):
    return {
        "ENABLE_OPENAI_API": request.app.state.config.ENABLE_OPENAI_API,
        "OPENAI_API_BASE_URLS": request.app.state.config.OPENAI_API_BASE_URLS,
        "OPENAI_API_KEYS": request.app.state.config.OPENAI_API_KEYS,
        "OPENAI_API_CONFIGS": request.app.state.config.OPENAI_API_CONFIGS,
    }


class OpenAIConfigForm(BaseModel):
    ENABLE_OPENAI_API: Optional[bool] = None
    OPENAI_API_BASE_URLS: list[str]
    OPENAI_API_KEYS: list[str]
    OPENAI_API_CONFIGS: dict


@router.post("/config/update")
async def update_config(
    request: Request, form_data: OpenAIConfigForm, user=Depends(get_admin_user)
):
    request.app.state.config.ENABLE_OPENAI_API = form_data.ENABLE_OPENAI_API
    request.app.state.config.OPENAI_API_BASE_URLS = form_data.OPENAI_API_BASE_URLS
    request.app.state.config.OPENAI_API_KEYS = form_data.OPENAI_API_KEYS

    # Check if API KEYS length is same than API URLS length
    if len(request.app.state.config.OPENAI_API_KEYS) != len(
        request.app.state.config.OPENAI_API_BASE_URLS
    ):
        if len(request.app.state.config.OPENAI_API_KEYS) > len(
            request.app.state.config.OPENAI_API_BASE_URLS
        ):
            request.app.state.config.OPENAI_API_KEYS = (
                request.app.state.config.OPENAI_API_KEYS[
                    : len(request.app.state.config.OPENAI_API_BASE_URLS)
                ]
            )
        else:
            request.app.state.config.OPENAI_API_KEYS += [""] * (
                len(request.app.state.config.OPENAI_API_BASE_URLS)
                - len(request.app.state.config.OPENAI_API_KEYS)
            )

    request.app.state.config.OPENAI_API_CONFIGS = form_data.OPENAI_API_CONFIGS

    # Remove the API configs that are not in the API URLS
    keys = list(map(str, range(len(request.app.state.config.OPENAI_API_BASE_URLS))))
    request.app.state.config.OPENAI_API_CONFIGS = {
        key: value
        for key, value in request.app.state.config.OPENAI_API_CONFIGS.items()
        if key in keys
    }

    return {
        "ENABLE_OPENAI_API": request.app.state.config.ENABLE_OPENAI_API,
        "OPENAI_API_BASE_URLS": request.app.state.config.OPENAI_API_BASE_URLS,
        "OPENAI_API_KEYS": request.app.state.config.OPENAI_API_KEYS,
        "OPENAI_API_CONFIGS": request.app.state.config.OPENAI_API_CONFIGS,
    }


@router.post("/audio/speech")
async def speech(request: Request, user=Depends(get_verified_user)):
    idx = None
    try:
        idx = request.app.state.config.OPENAI_API_BASE_URLS.index(
            "https://api.openai.com/v1"
        )

        body = await request.body()
        name = hashlib.sha256(body).hexdigest()

        SPEECH_CACHE_DIR = CACHE_DIR / "audio" / "speech"
        SPEECH_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        file_path = SPEECH_CACHE_DIR.joinpath(f"{name}.mp3")
        file_body_path = SPEECH_CACHE_DIR.joinpath(f"{name}.json")

        # Check if the file already exists in the cache
        if file_path.is_file():
            return FileResponse(file_path)

        url = request.app.state.config.OPENAI_API_BASE_URLS[idx]

        r = None
        try:
            r = requests.post(
                url=f"{url}/audio/speech",
                data=body,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {request.app.state.config.OPENAI_API_KEYS[idx]}",
                    **(
                        {
                            "HTTP-Referer": "https://openwebui.com/",
                            "X-Title": "Open WebUI",
                        }
                        if "openrouter.ai" in url
                        else {}
                    ),
                    **(
                        {
                            "X-OpenWebUI-User-Name": user.name,
                            "X-OpenWebUI-User-Id": user.id,
                            "X-OpenWebUI-User-Email": user.email,
                            "X-OpenWebUI-User-Role": user.role,
                        }
                        if ENABLE_FORWARD_USER_INFO_HEADERS
                        else {}
                    ),
                },
                stream=True,
            )

            r.raise_for_status()

            # Save the streaming content to a file
            with open(file_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

            with open(file_body_path, "w") as f:
                json.dump(json.loads(body.decode("utf-8")), f)

            # Return the saved file
            return FileResponse(file_path)

        except Exception as e:
            log.exception(e)

            detail = None
            if r is not None:
                try:
                    res = r.json()
                    if "error" in res:
                        detail = f"External: {res['error']}"
                except Exception:
                    detail = f"External: {e}"

            raise HTTPException(
                status_code=r.status_code if r else 500,
                detail=detail if detail else "Open WebUI: Server Connection Error",
            )

    except ValueError:
        raise HTTPException(status_code=401, detail=ERROR_MESSAGES.OPENAI_NOT_FOUND)


async def get_all_models_responses(request: Request, user: UserModel) -> list:
    if not request.app.state.config.ENABLE_OPENAI_API:
        return []

    # Check if API KEYS length is same than API URLS length
    num_urls = len(request.app.state.config.OPENAI_API_BASE_URLS)
    num_keys = len(request.app.state.config.OPENAI_API_KEYS)

    if num_keys != num_urls:
        # if there are more keys than urls, remove the extra keys
        if num_keys > num_urls:
            new_keys = request.app.state.config.OPENAI_API_KEYS[:num_urls]
            request.app.state.config.OPENAI_API_KEYS = new_keys
        # if there are more urls than keys, add empty keys
        else:
            request.app.state.config.OPENAI_API_KEYS += [""] * (num_urls - num_keys)

    request_tasks = []
    for idx, url in enumerate(request.app.state.config.OPENAI_API_BASE_URLS):
        if (str(idx) not in request.app.state.config.OPENAI_API_CONFIGS) and (
            url not in request.app.state.config.OPENAI_API_CONFIGS  # Legacy support
        ):
            request_tasks.append(
                send_get_request(
                    f"{url}/models",
                    request.app.state.config.OPENAI_API_KEYS[idx],
                    user=user,
                )
            )
        else:
            api_config = request.app.state.config.OPENAI_API_CONFIGS.get(
                str(idx),
                request.app.state.config.OPENAI_API_CONFIGS.get(
                    url, {}
                ),  # Legacy support
            )

            enable = api_config.get("enable", True)
            model_ids = api_config.get("model_ids", [])

            if enable:
                if len(model_ids) == 0:
                    request_tasks.append(
                        send_get_request(
                            f"{url}/models",
                            request.app.state.config.OPENAI_API_KEYS[idx],
                            user=user,
                        )
                    )
                else:
                    model_list = {
                        "object": "list",
                        "data": [
                            {
                                "id": model_id,
                                "name": model_id,
                                "owned_by": "openai",
                                "openai": {"id": model_id},
                                "urlIdx": idx,
                            }
                            for model_id in model_ids
                        ],
                    }

                    request_tasks.append(
                        asyncio.ensure_future(asyncio.sleep(0, model_list))
                    )
            else:
                request_tasks.append(asyncio.ensure_future(asyncio.sleep(0, None)))

    responses = await asyncio.gather(*request_tasks)

    for idx, response in enumerate(responses):
        if response:
            url = request.app.state.config.OPENAI_API_BASE_URLS[idx]
            api_config = request.app.state.config.OPENAI_API_CONFIGS.get(
                str(idx),
                request.app.state.config.OPENAI_API_CONFIGS.get(
                    url, {}
                ),  # Legacy support
            )

            connection_type = api_config.get("connection_type", "external")
            prefix_id = api_config.get("prefix_id", None)
            tags = api_config.get("tags", [])

            for model in (
                response if isinstance(response, list) else response.get("data", [])
            ):
                if prefix_id:
                    model["id"] = f"{prefix_id}.{model['id']}"

                if tags:
                    model["tags"] = tags

                if connection_type:
                    model["connection_type"] = connection_type

    log.debug(f"get_all_models:responses() {responses}")
    return responses


async def get_filtered_models(models, user):
    # Filter models based on user access control
    filtered_models = []
    for model in models.get("data", []):
        model_info = Models.get_model_by_id(model["id"])
        if model_info:
            if user.id == model_info.user_id or has_access(
                user.id, type="read", access_control=model_info.access_control
            ):
                filtered_models.append(model)
    return filtered_models


@cached(ttl=1)
async def get_all_models(request: Request, user: UserModel) -> dict[str, list]:
    log.info("get_all_models()")

    if not request.app.state.config.ENABLE_OPENAI_API:
        return {"data": []}

    responses = await get_all_models_responses(request, user=user)

    def extract_data(response):
        if response and "data" in response:
            return response["data"]
        if isinstance(response, list):
            return response
        return None

    def merge_models_lists(model_lists):
        log.debug(f"merge_models_lists {model_lists}")
        merged_list = []

        for idx, models in enumerate(model_lists):
            if models is not None and "error" not in models:

                merged_list.extend(
                    [
                        {
                            **model,
                            "name": model.get("name", model["id"]),
                            "owned_by": "openai",
                            "openai": model,
                            "connection_type": model.get("connection_type", "external"),
                            "urlIdx": idx,
                        }
                        for model in models
                        if (model.get("id") or model.get("name"))
                        and (
                            "api.openai.com"
                            not in request.app.state.config.OPENAI_API_BASE_URLS[idx]
                            or not any(
                                name in model["id"]
                                for name in [
                                    "babbage",
                                    "dall-e",
                                    "davinci",
                                    "embedding",
                                    "tts",
                                    "whisper",
                                ]
                            )
                        )
                    ]
                )

        return merged_list

    models = {"data": merge_models_lists(map(extract_data, responses))}
    log.debug(f"models: {models}")

    request.app.state.OPENAI_MODELS = {model["id"]: model for model in models["data"]}
    return models


@router.get("/models")
@router.get("/models/{url_idx}")
async def get_models(
    request: Request, url_idx: Optional[int] = None, user=Depends(get_verified_user)
):
    models = {
        "data": [],
    }

    if url_idx is None:
        models = await get_all_models(request, user=user)
    else:
        url = request.app.state.config.OPENAI_API_BASE_URLS[url_idx]
        key = request.app.state.config.OPENAI_API_KEYS[url_idx]

        api_config = request.app.state.config.OPENAI_API_CONFIGS.get(
            str(url_idx),
            request.app.state.config.OPENAI_API_CONFIGS.get(url, {}),  # Legacy support
        )

        r = None
        async with aiohttp.ClientSession(
            trust_env=True,
            timeout=aiohttp.ClientTimeout(total=AIOHTTP_CLIENT_TIMEOUT_MODEL_LIST),
        ) as session:
            try:
                headers = {
                    "Content-Type": "application/json",
                    **(
                        {
                            "X-OpenWebUI-User-Name": user.name,
                            "X-OpenWebUI-User-Id": user.id,
                            "X-OpenWebUI-User-Email": user.email,
                            "X-OpenWebUI-User-Role": user.role,
                        }
                        if ENABLE_FORWARD_USER_INFO_HEADERS
                        else {}
                    ),
                }

                if api_config.get("azure", False):
                    models = {
                        "data": api_config.get("model_ids", []) or [],
                        "object": "list",
                    }
                else:
                    headers["Authorization"] = f"Bearer {key}"

                    async with session.get(
                        f"{url}/models",
                        headers=headers,
                        ssl=AIOHTTP_CLIENT_SESSION_SSL,
                    ) as r:
                        if r.status != 200:
                            # Extract response error details if available
                            error_detail = f"HTTP Error: {r.status}"
                            res = await r.json()
                            if "error" in res:
                                error_detail = f"External Error: {res['error']}"
                            raise Exception(error_detail)

                        response_data = await r.json()

                        # Check if we're calling OpenAI API based on the URL
                        if "api.openai.com" in url:
                            # Filter models according to the specified conditions
                            response_data["data"] = [
                                model
                                for model in response_data.get("data", [])
                                if not any(
                                    name in model["id"]
                                    for name in [
                                        "babbage",
                                        "dall-e",
                                        "davinci",
                                        "embedding",
                                        "tts",
                                        "whisper",
                                    ]
                                )
                            ]

                        models = response_data
            except aiohttp.ClientError as e:
                # ClientError covers all aiohttp requests issues
                log.exception(f"Client error: {str(e)}")
                raise HTTPException(
                    status_code=500, detail="Open WebUI: Server Connection Error"
                )
            except Exception as e:
                log.exception(f"Unexpected error: {e}")
                error_detail = f"Unexpected error: {str(e)}"
                raise HTTPException(status_code=500, detail=error_detail)

    if user.role == "user" and not BYPASS_MODEL_ACCESS_CONTROL:
        models["data"] = await get_filtered_models(models, user)

    return models


class ConnectionVerificationForm(BaseModel):
    url: str
    key: str

    config: Optional[dict] = None


@router.post("/verify")
async def verify_connection(
    form_data: ConnectionVerificationForm, user=Depends(get_admin_user)
):
    url = form_data.url
    key = form_data.key

    api_config = form_data.config or {}

    async with aiohttp.ClientSession(
        trust_env=True,
        timeout=aiohttp.ClientTimeout(total=AIOHTTP_CLIENT_TIMEOUT_MODEL_LIST),
    ) as session:
        try:
            headers = {
                "Content-Type": "application/json",
                **(
                    {
                        "X-OpenWebUI-User-Name": user.name,
                        "X-OpenWebUI-User-Id": user.id,
                        "X-OpenWebUI-User-Email": user.email,
                        "X-OpenWebUI-User-Role": user.role,
                    }
                    if ENABLE_FORWARD_USER_INFO_HEADERS
                    else {}
                ),
            }

            if api_config.get("azure", False):
                headers["api-key"] = key
                api_version = api_config.get("api_version", "") or "2023-03-15-preview"

                async with session.get(
                    url=f"{url}/openai/models?api-version={api_version}",
                    headers=headers,
                    ssl=AIOHTTP_CLIENT_SESSION_SSL,
                ) as r:
                    if r.status != 200:
                        # Extract response error details if available
                        error_detail = f"HTTP Error: {r.status}"
                        res = await r.json()
                        if "error" in res:
                            error_detail = f"External Error: {res['error']}"
                        raise Exception(error_detail)

                    response_data = await r.json()
                    return response_data
            else:
                headers["Authorization"] = f"Bearer {key}"

                async with session.get(
                    f"{url}/models",
                    headers=headers,
                    ssl=AIOHTTP_CLIENT_SESSION_SSL,
                ) as r:
                    if r.status != 200:
                        # Extract response error details if available
                        error_detail = f"HTTP Error: {r.status}"
                        res = await r.json()
                        if "error" in res:
                            error_detail = f"External Error: {res['error']}"
                        raise Exception(error_detail)

                    response_data = await r.json()
                    return response_data

        except aiohttp.ClientError as e:
            # ClientError covers all aiohttp requests issues
            log.exception(f"Client error: {str(e)}")
            raise HTTPException(
                status_code=500, detail="Open WebUI: Server Connection Error"
            )
        except Exception as e:
            log.exception(f"Unexpected error: {e}")
            error_detail = f"Unexpected error: {str(e)}"
            raise HTTPException(status_code=500, detail=error_detail)


def convert_to_azure_payload(
    url,
    payload: dict,
):
    model = payload.get("model", "")

    # Filter allowed parameters based on Azure OpenAI API
    allowed_params = {
        "messages",
        "temperature",
        "role",
        "content",
        "contentPart",
        "contentPartImage",
        "enhancements",
        "dataSources",
        "n",
        "stream",
        "stop",
        "max_tokens",
        "presence_penalty",
        "frequency_penalty",
        "logit_bias",
        "user",
        "function_call",
        "functions",
        "tools",
        "tool_choice",
        "top_p",
        "log_probs",
        "top_logprobs",
        "response_format",
        "seed",
        "max_completion_tokens",
    }

    # Special handling for o-series models
    if model.startswith("o") and model.endswith("-mini"):
        # Convert max_tokens to max_completion_tokens for o-series models
        if "max_tokens" in payload:
            payload["max_completion_tokens"] = payload["max_tokens"]
            del payload["max_tokens"]

        # Remove temperature if not 1 for o-series models
        if "temperature" in payload and payload["temperature"] != 1:
            log.debug(
                f"Removing temperature parameter for o-series model {model} as only default value (1) is supported"
            )
            del payload["temperature"]

    # Filter out unsupported parameters
    payload = {k: v for k, v in payload.items() if k in allowed_params}

    url = f"{url}/openai/deployments/{model}"
    return url, payload


@router.post("/chat/completions")
async def oci_generate_chat_completion(
    request: Request,
    form_data: dict,
    user=Depends(get_verified_user),
):

    breakpoint()
    print("\n\n==== INICIANDO ENDPOINT /chat/completions ====")
    print(f"Form data recebido: {form_data}")
    
    # Criar um usuário fictício para testes com todos os campos necessários
    import time
    from open_webui.models.users import UserModel, UserSettings
    current_time = int(time.time())
    user = UserModel(
        id="test-user", 
        email="test@example.com", 
        name="Test User", 
        role="admin",  # Definindo como admin para garantir acesso a todos os modelos
        profile_image_url="https://example.com/profile.jpg",
        last_active_at=current_time,
        updated_at=current_time,
        created_at=current_time,
        settings=UserSettings(ui={}),
        info={}
    )

    # Forçar bypass_filter para True para desativar verificações de acesso ao modelo
    bypass_filter = True

    idx = 0

    # Extrair os dados do form_data
    model_id = form_data.get("model")
    messages_data = form_data.get("messages", [])
    stream = form_data.get("stream", False)
    
    print(f"Modelo selecionado: {model_id}")
    print(f"Mensagens recebidas: {messages_data}")
    
    # Converter as mensagens do formato dict para objetos Message
    messages = []
    for msg_data in messages_data:
        msg = Message(
            role=msg_data.get("role"),
            content=msg_data.get("content")
        )
        messages.append(msg)
        print(f"Mensagem processada: {msg.role} - {msg.content[:30]}...")
    
    # Criar o objeto ChatCompletionRequest
    chat_request = ChatCompletionRequest(
        model=model_id,
        messages=messages,
        stream=True
    )
    
    print(f"ChatCompletionRequest criado: {chat_request}")
    
    # Chamar a função de completions com o objeto criado
    return await chat_completation_3(chat_request)
    


async def embeddings(request: Request, form_data: dict, user):
    """
    Calls the embeddings endpoint for OpenAI-compatible providers.

    Args:
        request (Request): The FastAPI request context.
        form_data (dict): OpenAI-compatible embeddings payload.
        user (UserModel): The authenticated user.

    Returns:
        dict: OpenAI-compatible embeddings response.
    """
    idx = 0
    # Prepare payload/body
    body = json.dumps(form_data)
    # Find correct backend url/key based on model
    await get_all_models(request, user=user)
    model_id = form_data.get("model")
    models = request.app.state.OPENAI_MODELS
    if model_id in models:
        idx = models[model_id]["urlIdx"]
    url = request.app.state.config.OPENAI_API_BASE_URLS[idx]
    key = request.app.state.config.OPENAI_API_KEYS[idx]
    r = None
    session = None
    streaming = False
    try:
        session = aiohttp.ClientSession(trust_env=True)
        r = await session.request(
            method="POST",
            url=f"{url}/embeddings",
            data=body,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
                **(
                    {
                        "X-OpenWebUI-User-Name": user.name,
                        "X-OpenWebUI-User-Id": user.id,
                        "X-OpenWebUI-User-Email": user.email,
                        "X-OpenWebUI-User-Role": user.role,
                    }
                    if ENABLE_FORWARD_USER_INFO_HEADERS and user
                    else {}
                ),
            },
        )
        r.raise_for_status()
        if "text/event-stream" in r.headers.get("Content-Type", ""):
            streaming = True
            return StreamingResponse(
                r.content,
                status_code=r.status,
                headers=dict(r.headers),
                background=BackgroundTask(
                    cleanup_response, response=r, session=session
                ),
            )
        else:
            response_data = await r.json()
            return response_data
    except Exception as e:
        log.exception(e)
        detail = None
        if r is not None:
            try:
                res = await r.json()
                if "error" in res:
                    detail = f"External: {res['error']['message'] if 'message' in res['error'] else res['error']}"
            except Exception:
                detail = f"External: {e}"
        raise HTTPException(
            status_code=r.status if r else 500,
            detail=detail if detail else "Open WebUI: Server Connection Error",
        )
    finally:
        if not streaming and session:
            if r:
                r.close()
            await session.close()


@router.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy(path: str, request: Request, user=Depends(get_verified_user)):
    """
    Deprecated: proxy all requests to OpenAI API
    """

    body = await request.body()

    idx = 0
    url = request.app.state.config.OPENAI_API_BASE_URLS[idx]
    key = request.app.state.config.OPENAI_API_KEYS[idx]
    api_config = request.app.state.config.OPENAI_API_CONFIGS.get(
        str(idx),
        request.app.state.config.OPENAI_API_CONFIGS.get(
            request.app.state.config.OPENAI_API_BASE_URLS[idx], {}
        ),  # Legacy support
    )

    r = None
    session = None
    streaming = False

    try:
        headers = {
            "Content-Type": "application/json",
            **(
                {
                    "X-OpenWebUI-User-Name": user.name,
                    "X-OpenWebUI-User-Id": user.id,
                    "X-OpenWebUI-User-Email": user.email,
                    "X-OpenWebUI-User-Role": user.role,
                }
                if ENABLE_FORWARD_USER_INFO_HEADERS
                else {}
            ),
        }

        if api_config.get("azure", False):
            headers["api-key"] = key
            headers["api-version"] = (
                api_config.get("api_version", "") or "2023-03-15-preview"
            )

            payload = json.loads(body)
            url, payload = convert_to_azure_payload(url, payload)
            body = json.dumps(payload).encode()

            request_url = f"{url}/{path}?api-version={api_config.get('api_version', '2023-03-15-preview')}"
        else:
            headers["Authorization"] = f"Bearer {key}"
            request_url = f"{url}/{path}"

        session = aiohttp.ClientSession(trust_env=True)
        r = await session.request(
            method=request.method,
            url=request_url,
            data=body,
            headers=headers,
            ssl=AIOHTTP_CLIENT_SESSION_SSL,
        )
        r.raise_for_status()

        # Check if response is SSE
        if "text/event-stream" in r.headers.get("Content-Type", ""):
            streaming = True
            return StreamingResponse(
                r.content,
                status_code=r.status,
                headers=dict(r.headers),
                background=BackgroundTask(
                    cleanup_response, response=r, session=session
                ),
            )
        else:
            response_data = await r.json()
            return response_data

    except Exception as e:
        log.exception(e)

        detail = None
        if r is not None:
            try:
                res = await r.json()
                log.error(res)
                if "error" in res:
                    detail = f"External: {res['error']['message'] if 'message' in res['error'] else res['error']}"
            except Exception:
                detail = f"External: {e}"
        raise HTTPException(
            status_code=r.status if r else 500,
            detail=detail if detail else "Open WebUI: Server Connection Error",
        )
    finally:
        if not streaming and session:
            if r:
                r.close()
            await session.close()
