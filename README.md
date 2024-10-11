<div class="Box-sc-g0xbh4-0 QkQOb js-snippet-clipboard-copy-unpositioned" data-hpc="true"><article class="markdown-body entry-content container-lg" itemprop="text"><div class="markdown-heading" dir="auto"><h1 tabindex="-1" class="heading-element" dir="auto" _msttexthash="29271138" _msthash="200">llama3 从头开始实现</h1><a id="user-content-llama3-implemented-from-scratch" class="anchor" aria-label="永久链接：llama3 从头开始实现" href="#llama3-implemented-from-scratch" _mstaria-label="1312766" _msthash="201"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></div>
<p dir="auto" _msttexthash="1634520472" _msthash="202">在这个文件中，我从头开始实现了 LLAMA3，一次一个张量和矩阵乘法。<br _istranslated="1">此外，我要直接从 meta 为 LLAMA3 提供的模型文件加载张量，您需要在运行此文件之前下载权重。
这是下载权重的官方链接：<a href="https://llama.meta.com/llama-downloads/" rel="nofollow" _istranslated="1">https://llama.meta.com/llama-downloads/</a></p>
<div dir="auto">
    <a target="_blank" rel="noopener noreferrer" href="https://github.com/naklecha/llama3-from-scratch/blob/main/images/archi.png"><img src="https://github.com/naklecha/llama3-from-scratch/raw/main/images/archi.png" style="max-width: 100%;"></a>
</div>
<div class="markdown-heading" dir="auto"><h2 tabindex="-1" class="heading-element" dir="auto" _msttexthash="8220914" _msthash="203">分词器</h2><a id="user-content-tokenizer" class="anchor" aria-label="永久链接：分词器" href="#tokenizer" _mstaria-label="416104" _msthash="204"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></div>
<p dir="auto" _msttexthash="397407920" _msthash="205">我不打算实现 BPE 分词器（但 Andrej Karpathy 有一个非常干净的实现）<br _istranslated="1">链接到他的实现：<a href="https://github.com/karpathy/minbpe" _istranslated="1">https://github.com/karpathy/minbpe</a></p>
<div dir="auto">
    <a target="_blank" rel="noopener noreferrer" href="https://github.com/naklecha/llama3-from-scratch/blob/main/images/karpathyminbpe.png"><img src="https://github.com/naklecha/llama3-from-scratch/raw/main/images/karpathyminbpe.png" width="600" style="max-width: 100%;"></a>
</div>
<div class="highlight highlight-source-python notranslate position-relative overflow-auto" dir="auto"><pre><span class="pl-k">from</span> <span class="pl-s1">pathlib</span> <span class="pl-k">import</span> <span class="pl-v">Path</span>
<span class="pl-k">import</span> <span class="pl-s1">tiktoken</span>
<span class="pl-k">from</span> <span class="pl-s1">tiktoken</span>.<span class="pl-s1">load</span> <span class="pl-k">import</span> <span class="pl-s1">load_tiktoken_bpe</span>
<span class="pl-k">import</span> <span class="pl-s1">torch</span>
<span class="pl-k">import</span> <span class="pl-s1">json</span>
<span class="pl-k">import</span> <span class="pl-s1">matplotlib</span>.<span class="pl-s1">pyplot</span> <span class="pl-k">as</span> <span class="pl-s1">plt</span>

<span class="pl-s1">tokenizer_path</span> <span class="pl-c1">=</span> <span class="pl-s">"Meta-Llama-3-8B/tokenizer.model"</span>
<span class="pl-s1">special_tokens</span> <span class="pl-c1">=</span> [
            <span class="pl-s">"&lt;|begin_of_text|&gt;"</span>,
            <span class="pl-s">"&lt;|end_of_text|&gt;"</span>,
            <span class="pl-s">"&lt;|reserved_special_token_0|&gt;"</span>,
            <span class="pl-s">"&lt;|reserved_special_token_1|&gt;"</span>,
            <span class="pl-s">"&lt;|reserved_special_token_2|&gt;"</span>,
            <span class="pl-s">"&lt;|reserved_special_token_3|&gt;"</span>,
            <span class="pl-s">"&lt;|start_header_id|&gt;"</span>,
            <span class="pl-s">"&lt;|end_header_id|&gt;"</span>,
            <span class="pl-s">"&lt;|reserved_special_token_4|&gt;"</span>,
            <span class="pl-s">"&lt;|eot_id|&gt;"</span>,  <span class="pl-c"># end of turn</span>
        ] <span class="pl-c1">+</span> [<span class="pl-s">f"&lt;|reserved_special_token_<span class="pl-s1"><span class="pl-kos">{</span><span class="pl-s1">i</span><span class="pl-kos">}</span></span>|&gt;"</span> <span class="pl-k">for</span> <span class="pl-s1">i</span> <span class="pl-c1">in</span> <span class="pl-en">range</span>(<span class="pl-c1">5</span>, <span class="pl-c1">256</span> <span class="pl-c1">-</span> <span class="pl-c1">5</span>)]
<span class="pl-s1">mergeable_ranks</span> <span class="pl-c1">=</span> <span class="pl-en">load_tiktoken_bpe</span>(<span class="pl-s1">tokenizer_path</span>)
<span class="pl-s1">tokenizer</span> <span class="pl-c1">=</span> <span class="pl-s1">tiktoken</span>.<span class="pl-v">Encoding</span>(
    <span class="pl-s1">name</span><span class="pl-c1">=</span><span class="pl-v">Path</span>(<span class="pl-s1">tokenizer_path</span>).<span class="pl-s1">name</span>,
    <span class="pl-s1">pat_str</span><span class="pl-c1">=</span><span class="pl-s">r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"</span>,
    <span class="pl-s1">mergeable_ranks</span><span class="pl-c1">=</span><span class="pl-s1">mergeable_ranks</span>,
    <span class="pl-s1">special_tokens</span><span class="pl-c1">=</span>{<span class="pl-s1">token</span>: <span class="pl-en">len</span>(<span class="pl-s1">mergeable_ranks</span>) <span class="pl-c1">+</span> <span class="pl-s1">i</span> <span class="pl-k">for</span> <span class="pl-s1">i</span>, <span class="pl-s1">token</span> <span class="pl-c1">in</span> <span class="pl-en">enumerate</span>(<span class="pl-s1">special_tokens</span>)},
)

<span class="pl-s1">tokenizer</span>.<span class="pl-en">decode</span>(<span class="pl-s1">tokenizer</span>.<span class="pl-en">encode</span>(<span class="pl-s">"hello world!"</span>))</pre><div class="zeroclipboard-container">
    
  </div></div>
<div class="snippet-clipboard-content notranslate position-relative overflow-auto"><pre class="notranslate"><code>'hello world!'
</code></pre><div class="zeroclipboard-container">
   
  </div></div>
<div class="markdown-heading" dir="auto"><h2 tabindex="-1" class="heading-element" dir="auto" _msttexthash="18455333" _msthash="206">读取模型文件</h2><a id="user-content-reading-the-model-file" class="anchor" aria-label="永久链接：读取模型文件" href="#reading-the-model-file" _mstaria-label="820989" _msthash="207"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></div>
<p dir="auto" _msttexthash="664167673" _msthash="208">通常，阅读此内容取决于模型类的编写方式以及其中的变量名称。<br _istranslated="1">但是由于我们从头开始实现 LLAMA3，因此我们将一次读取一个 Tensor 文件。</p>
<div dir="auto">
    <a target="_blank" rel="noopener noreferrer" href="https://github.com/naklecha/llama3-from-scratch/blob/main/images/model.png"><img src="https://github.com/naklecha/llama3-from-scratch/raw/main/images/model.png" width="600" style="max-width: 100%;"></a>
</div>
<div class="highlight highlight-source-python notranslate position-relative overflow-auto" dir="auto"><pre><span class="pl-s1">model</span> <span class="pl-c1">=</span> <span class="pl-s1">torch</span>.<span class="pl-en">load</span>(<span class="pl-s">"Meta-Llama-3-8B/consolidated.00.pth"</span>)
<span class="pl-en">print</span>(<span class="pl-s1">json</span>.<span class="pl-en">dumps</span>(<span class="pl-en">list</span>(<span class="pl-s1">model</span>.<span class="pl-en">keys</span>())[:<span class="pl-c1">20</span>], <span class="pl-s1">indent</span><span class="pl-c1">=</span><span class="pl-c1">4</span>))</pre><div class="zeroclipboard-container">
   
  </div></div>
<div class="snippet-clipboard-content notranslate position-relative overflow-auto"><pre class="notranslate"><code>[
    "tok_embeddings.weight",
    "layers.0.attention.wq.weight",
    "layers.0.attention.wk.weight",
    "layers.0.attention.wv.weight",
    "layers.0.attention.wo.weight",
    "layers.0.feed_forward.w1.weight",
    "layers.0.feed_forward.w3.weight",
    "layers.0.feed_forward.w2.weight",
    "layers.0.attention_norm.weight",
    "layers.0.ffn_norm.weight",
    "layers.1.attention.wq.weight",
    "layers.1.attention.wk.weight",
    "layers.1.attention.wv.weight",
    "layers.1.attention.wo.weight",
    "layers.1.feed_forward.w1.weight",
    "layers.1.feed_forward.w3.weight",
    "layers.1.feed_forward.w2.weight",
    "layers.1.attention_norm.weight",
    "layers.1.ffn_norm.weight",
    "layers.2.attention.wq.weight"
]
</code></pre><div class="zeroclipboard-container">
     
  </div></div>
<div class="highlight highlight-source-python notranslate position-relative overflow-auto" dir="auto"><pre><span class="pl-k">with</span> <span class="pl-en">open</span>(<span class="pl-s">"Meta-Llama-3-8B/params.json"</span>, <span class="pl-s">"r"</span>) <span class="pl-k">as</span> <span class="pl-s1">f</span>:
    <span class="pl-s1">config</span> <span class="pl-c1">=</span> <span class="pl-s1">json</span>.<span class="pl-en">load</span>(<span class="pl-s1">f</span>)
<span class="pl-s1">config</span></pre><div class="zeroclipboard-container">
    
  </div></div>
<div class="snippet-clipboard-content notranslate position-relative overflow-auto"><pre class="notranslate"><code>{'dim': 4096,
 'n_layers': 32,
 'n_heads': 32,
 'n_kv_heads': 8,
 'vocab_size': 128256,
 'multiple_of': 1024,
 'ffn_dim_multiplier': 1.3,
 'norm_eps': 1e-05,
 'rope_theta': 500000.0}
</code></pre><div class="zeroclipboard-container">
    
  </div></div>
<div class="markdown-heading" dir="auto"><h2 tabindex="-1" class="heading-element" dir="auto" _msttexthash="126653553" _msthash="209">我们使用此配置来推断有关模型的细节，例如</h2><a id="user-content-we-use-this-config-to-infer-details-about-the-model-like" class="anchor" aria-label="永久链接：我们使用这个配置来推断有关模型的细节，例如" href="#we-use-this-config-to-infer-details-about-the-model-like" _mstaria-label="2671344" _msthash="210"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></div>
<ol dir="auto">
<li _msttexthash="33429890" _msthash="211">该模型有 32 个变压器层</li>
<li _msttexthash="41224157" _msthash="212">每个多头注意力块有 32 个头</li>
<li _msttexthash="16386799" _msthash="213">词汇大小等</li>
</ol>
<div class="highlight highlight-source-python notranslate position-relative overflow-auto" dir="auto"><pre><span class="pl-s1">dim</span> <span class="pl-c1">=</span> <span class="pl-s1">config</span>[<span class="pl-s">"dim"</span>]
<span class="pl-s1">n_layers</span> <span class="pl-c1">=</span> <span class="pl-s1">config</span>[<span class="pl-s">"n_layers"</span>]
<span class="pl-s1">n_heads</span> <span class="pl-c1">=</span> <span class="pl-s1">config</span>[<span class="pl-s">"n_heads"</span>]
<span class="pl-s1">n_kv_heads</span> <span class="pl-c1">=</span> <span class="pl-s1">config</span>[<span class="pl-s">"n_kv_heads"</span>]
<span class="pl-s1">vocab_size</span> <span class="pl-c1">=</span> <span class="pl-s1">config</span>[<span class="pl-s">"vocab_size"</span>]
<span class="pl-s1">multiple_of</span> <span class="pl-c1">=</span> <span class="pl-s1">config</span>[<span class="pl-s">"multiple_of"</span>]
<span class="pl-s1">ffn_dim_multiplier</span> <span class="pl-c1">=</span> <span class="pl-s1">config</span>[<span class="pl-s">"ffn_dim_multiplier"</span>]
<span class="pl-s1">norm_eps</span> <span class="pl-c1">=</span> <span class="pl-s1">config</span>[<span class="pl-s">"norm_eps"</span>]
<span class="pl-s1">rope_theta</span> <span class="pl-c1">=</span> <span class="pl-s1">torch</span>.<span class="pl-en">tensor</span>(<span class="pl-s1">config</span>[<span class="pl-s">"rope_theta"</span>])</pre><div class="zeroclipboard-container">
     
  </div></div>
<div class="markdown-heading" dir="auto"><h2 tabindex="-1" class="heading-element" dir="auto" _msttexthash="30481347" _msthash="214">将文本转换为标记</h2><a id="user-content-converting-text-to-tokens" class="anchor" aria-label="固定链接：将文本转换为标记" href="#converting-text-to-tokens" _mstaria-label="1015976" _msthash="215"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></div>
<p dir="auto" _msttexthash="209508403" _msthash="216">在这里，我们使用 TikToken（我认为是一个 OpenAI 库）作为分词器</p>
<div dir="auto">
    <a target="_blank" rel="noopener noreferrer" href="https://github.com/naklecha/llama3-from-scratch/blob/main/images/tokens.png"><img src="https://github.com/naklecha/llama3-from-scratch/raw/main/images/tokens.png" width="600" style="max-width: 100%;"></a>
</div>
<div class="highlight highlight-source-python notranslate position-relative overflow-auto" dir="auto"><pre><span class="pl-s1">prompt</span> <span class="pl-c1">=</span> <span class="pl-s">"the answer to the ultimate question of life, the universe, and everything is "</span>
<span class="pl-s1">tokens</span> <span class="pl-c1">=</span> [<span class="pl-c1">128000</span>] <span class="pl-c1">+</span> <span class="pl-s1">tokenizer</span>.<span class="pl-en">encode</span>(<span class="pl-s1">prompt</span>)
<span class="pl-en">print</span>(<span class="pl-s1">tokens</span>)
<span class="pl-s1">tokens</span> <span class="pl-c1">=</span> <span class="pl-s1">torch</span>.<span class="pl-en">tensor</span>(<span class="pl-s1">tokens</span>)
<span class="pl-s1">prompt_split_as_tokens</span> <span class="pl-c1">=</span> [<span class="pl-s1">tokenizer</span>.<span class="pl-en">decode</span>([<span class="pl-s1">token</span>.<span class="pl-en">item</span>()]) <span class="pl-k">for</span> <span class="pl-s1">token</span> <span class="pl-c1">in</span> <span class="pl-s1">tokens</span>]
<span class="pl-en">print</span>(<span class="pl-s1">prompt_split_as_tokens</span>)</pre><div class="zeroclipboard-container">
   
  </div></div>
<div class="snippet-clipboard-content notranslate position-relative overflow-auto"><pre class="notranslate"><code>[128000, 1820, 4320, 311, 279, 17139, 3488, 315, 2324, 11, 279, 15861, 11, 323, 4395, 374, 220]
['&lt;|begin_of_text|&gt;', 'the', ' answer', ' to', ' the', ' ultimate', ' question', ' of', ' life', ',', ' the', ' universe', ',', ' and', ' everything', ' is', ' ']
</code></pre><div class="zeroclipboard-container">
    
  </div></div>
<div class="markdown-heading" dir="auto"><h2 tabindex="-1" class="heading-element" dir="auto" _msttexthash="31443386" _msthash="217">将 Token 转换为其嵌入</h2><a id="user-content-converting-tokens-to-their-embedding" class="anchor" aria-label="永久链接：将 token 转换为其嵌入" href="#converting-tokens-to-their-embedding" _mstaria-label="1592695" _msthash="218"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></div>
<p dir="auto" _msttexthash="1591982977" _msthash="219">对不起，这是代码库中唯一使用内置神经网络模块<br _istranslated="1">的部分，所以我们的 [17x1] 令牌现在是 [17x4096]，即 17 个长度为 4096 <br _istranslated="1"> 的嵌入（每个令牌一个）<br _istranslated="1">注意：跟踪形状，它使理解所有内容变得更加容易</p>
<div dir="auto">
    <a target="_blank" rel="noopener noreferrer" href="https://github.com/naklecha/llama3-from-scratch/blob/main/images/embeddings.png"><img src="https://github.com/naklecha/llama3-from-scratch/raw/main/images/embeddings.png" width="600" style="max-width: 100%;"></a>
</div>
<div class="highlight highlight-source-python notranslate position-relative overflow-auto" dir="auto"><pre><span class="pl-s1">embedding_layer</span> <span class="pl-c1">=</span> <span class="pl-s1">torch</span>.<span class="pl-s1">nn</span>.<span class="pl-v">Embedding</span>(<span class="pl-s1">vocab_size</span>, <span class="pl-s1">dim</span>)
<span class="pl-s1">embedding_layer</span>.<span class="pl-s1">weight</span>.<span class="pl-s1">data</span>.<span class="pl-en">copy_</span>(<span class="pl-s1">model</span>[<span class="pl-s">"tok_embeddings.weight"</span>])
<span class="pl-s1">token_embeddings_unnormalized</span> <span class="pl-c1">=</span> <span class="pl-en">embedding_layer</span>(<span class="pl-s1">tokens</span>).<span class="pl-en">to</span>(<span class="pl-s1">torch</span>.<span class="pl-s1">bfloat16</span>)
<span class="pl-s1">token_embeddings_unnormalized</span>.<span class="pl-s1">shape</span></pre><div class="zeroclipboard-container">
    
  </div></div>
<div class="snippet-clipboard-content notranslate position-relative overflow-auto"><pre class="notranslate"><code>torch.Size([17, 4096])
</code></pre><div class="zeroclipboard-container">
    
  </div></div>
<div class="markdown-heading" dir="auto"><h2 tabindex="-1" class="heading-element" dir="auto" _msttexthash="104921466" _msthash="220">然后，我们使用 RMS 归一化对嵌入进行归一化</h2><a id="user-content-we-then-normalize-the-embedding-using-rms-normalization" class="anchor" aria-label="永久链接：然后我们使用 rms 归一化对嵌入进行归一化" href="#we-then-normalize-the-embedding-using-rms-normalization" _mstaria-label="2876900" _msthash="221"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></div>
<p dir="auto" _msttexthash="1163759649" _msthash="222">请注意，在此步骤之后，形状不会改变，值只是要记住的标准化<br _istranslated="1">事情，我们需要一个norm_eps（来自 config），因为我们不想意外地将 rms 设置为 0 并除以 0，<br _istranslated="1">公式如下：</p>
<div dir="auto">
    <a target="_blank" rel="noopener noreferrer" href="https://github.com/naklecha/llama3-from-scratch/blob/main/images/rms.png"><img src="https://github.com/naklecha/llama3-from-scratch/raw/main/images/rms.png" width="600" style="max-width: 100%;"></a>
</div>
<div class="highlight highlight-source-python notranslate position-relative overflow-auto" dir="auto"><pre><span class="pl-c"># def rms_norm(tensor, norm_weights):</span>
<span class="pl-c">#     rms = (tensor.pow(2).mean(-1, keepdim=True) + norm_eps)**0.5</span>
<span class="pl-c">#     return tensor * (norm_weights / rms)</span>
<span class="pl-k">def</span> <span class="pl-en">rms_norm</span>(<span class="pl-s1">tensor</span>, <span class="pl-s1">norm_weights</span>):
    <span class="pl-k">return</span> (<span class="pl-s1">tensor</span> <span class="pl-c1">*</span> <span class="pl-s1">torch</span>.<span class="pl-en">rsqrt</span>(<span class="pl-s1">tensor</span>.<span class="pl-en">pow</span>(<span class="pl-c1">2</span>).<span class="pl-en">mean</span>(<span class="pl-c1">-</span><span class="pl-c1">1</span>, <span class="pl-s1">keepdim</span><span class="pl-c1">=</span><span class="pl-c1">True</span>) <span class="pl-c1">+</span> <span class="pl-s1">norm_eps</span>)) <span class="pl-c1">*</span> <span class="pl-s1">norm_weights</span></pre><div class="zeroclipboard-container">
    
  </div></div>
<div class="markdown-heading" dir="auto"><h1 tabindex="-1" class="heading-element" dir="auto" _msttexthash="34428082" _msthash="223">构建 transformer 的第一层</h1><a id="user-content-building-the-first-first-layer-of-the-transformer" class="anchor" aria-label="永久链接：构建 transformer 的第一层" href="#building-the-first-first-layer-of-the-transformer" _mstaria-label="2370108" _msthash="224"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></div>
<div class="markdown-heading" dir="auto"><h3 tabindex="-1" class="heading-element" dir="auto" _msttexthash="7498751" _msthash="225">正常化</h3><a id="user-content-normalization" class="anchor" aria-label="永久链接：规范化" href="#normalization" _mstaria-label="569777" _msthash="226"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></div>
<p dir="auto" _msttexthash="816538320" _msthash="227">无论如何，你会看到我从模型 dict 中访问 layer.0（这是第一层），<br _istranslated="1">所以在规范化后，我们的形状仍然 [17x4096] 与嵌入相同，但已规范化</p>
<div dir="auto">
    <a target="_blank" rel="noopener noreferrer" href="https://github.com/naklecha/llama3-from-scratch/blob/main/images/norm.png"><img src="https://github.com/naklecha/llama3-from-scratch/raw/main/images/norm.png" width="600" style="max-width: 100%;"></a>
</div>
<div class="highlight highlight-source-python notranslate position-relative overflow-auto" dir="auto"><pre><span class="pl-s1">token_embeddings</span> <span class="pl-c1">=</span> <span class="pl-en">rms_norm</span>(<span class="pl-s1">token_embeddings_unnormalized</span>, <span class="pl-s1">model</span>[<span class="pl-s">"layers.0.attention_norm.weight"</span>])
<span class="pl-s1">token_embeddings</span>.<span class="pl-s1">shape</span></pre><div class="zeroclipboard-container">
    
  </div></div>
<div class="snippet-clipboard-content notranslate position-relative overflow-auto"><pre class="notranslate"><code>torch.Size([17, 4096])
</code></pre><div class="zeroclipboard-container">
    
  </div></div>
<div class="markdown-heading" dir="auto"><h3 tabindex="-1" class="heading-element" dir="auto" _msttexthash="24457524" _msthash="228">从零开始实施的 Attention</h3><a id="user-content-attention-implemented-from-scratch" class="anchor" aria-label="永久链接：从零开始实施注意力" href="#attention-implemented-from-scratch" _mstaria-label="1530464" _msthash="229"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></div>
<p dir="auto" _msttexthash="50737128" _msthash="230">让我们加载 transformer 第一层的 attention heads</p>
<div dir="auto">
    <a target="_blank" rel="noopener noreferrer" href="https://github.com/naklecha/llama3-from-scratch/blob/main/images/qkv.png"><img src="https://github.com/naklecha/llama3-from-scratch/raw/main/images/qkv.png" width="600" style="max-width: 100%;"></a>
</div>
<br>
<p dir="auto" _msttexthash="3459772017" _msthash="231">&gt;当我们从模型中加载查询、键、值和输出向量时，我们注意到形状是 [4096x4096]、[1024x4096]、[1024x4096]、[4096x4096] <br _istranslated="1"> &gt;乍一看这很奇怪，因为理想情况下，我们希望每个头的每个 q、k、v 和 o 单独&gt;<br _istranslated="1">代码的作者将它们捆绑在一起，因为它很容易，它有助于比较注意力头乘法。<br _istranslated="1"> &gt;我要解开所有东西......</p>
<div class="highlight highlight-source-python notranslate position-relative overflow-auto" dir="auto"><pre><span class="pl-en">print</span>(
    <span class="pl-s1">model</span>[<span class="pl-s">"layers.0.attention.wq.weight"</span>].<span class="pl-s1">shape</span>,
    <span class="pl-s1">model</span>[<span class="pl-s">"layers.0.attention.wk.weight"</span>].<span class="pl-s1">shape</span>,
    <span class="pl-s1">model</span>[<span class="pl-s">"layers.0.attention.wv.weight"</span>].<span class="pl-s1">shape</span>,
    <span class="pl-s1">model</span>[<span class="pl-s">"layers.0.attention.wo.weight"</span>].<span class="pl-s1">shape</span>
)</pre><div class="zeroclipboard-container">
    
  </div></div>
<div class="snippet-clipboard-content notranslate position-relative overflow-auto"><pre class="notranslate"><code>torch.Size([4096, 4096]) torch.Size([1024, 4096]) torch.Size([1024, 4096]) torch.Size([4096, 4096])
</code></pre><div class="zeroclipboard-container">
   
  </div></div>
<div class="markdown-heading" dir="auto"><h3 tabindex="-1" class="heading-element" dir="auto" _msttexthash="13189670" _msthash="232">解包查询</h3><a id="user-content-unwrapping-query" class="anchor" aria-label="固定链接：解包查询" href="#unwrapping-query" _mstaria-label="666211" _msthash="233"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></div>
<p dir="auto" _msttexthash="1137650384" _msthash="234">在下一节中，我们将解包来自多个 Attention Heads 的查询，结果形状为 [32x128x4096]<br _istranslated="1"><br _istranslated="1">，32 是 llama3 中的注意力头数量，128 是查询向量的大小，4096 是标记嵌入的大小</p>
<div class="highlight highlight-source-python notranslate position-relative overflow-auto" dir="auto"><pre><span class="pl-s1">q_layer0</span> <span class="pl-c1">=</span> <span class="pl-s1">model</span>[<span class="pl-s">"layers.0.attention.wq.weight"</span>]
<span class="pl-s1">head_dim</span> <span class="pl-c1">=</span> <span class="pl-s1">q_layer0</span>.<span class="pl-s1">shape</span>[<span class="pl-c1">0</span>] <span class="pl-c1">//</span> <span class="pl-s1">n_heads</span>
<span class="pl-s1">q_layer0</span> <span class="pl-c1">=</span> <span class="pl-s1">q_layer0</span>.<span class="pl-en">view</span>(<span class="pl-s1">n_heads</span>, <span class="pl-s1">head_dim</span>, <span class="pl-s1">dim</span>)
<span class="pl-s1">q_layer0</span>.<span class="pl-s1">shape</span></pre><div class="zeroclipboard-container">
    
  </div></div>
<div class="snippet-clipboard-content notranslate position-relative overflow-auto"><pre class="notranslate"><code>torch.Size([32, 128, 4096])
</code></pre><div class="zeroclipboard-container">
     
  </div></div>
<div class="markdown-heading" dir="auto"><h3 tabindex="-1" class="heading-element" dir="auto" _msttexthash="43292288" _msthash="235">我将实现第一层的第一个 head</h3><a id="user-content-im-going-to-implement-the-first-head-of-the-first-layer" class="anchor" aria-label="永久链接：我要实现第一层的第一个头" href="#im-going-to-implement-the-first-head-of-the-first-layer" _mstaria-label="2605031" _msthash="236"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></div>
<p dir="auto" _msttexthash="316914091" _msthash="237">这里我访问的是第一层的查询权重矩阵第一个头，这个查询权重矩阵的大小是 [128x4096]</p>
<div class="highlight highlight-source-python notranslate position-relative overflow-auto" dir="auto"><pre><span class="pl-s1">q_layer0_head0</span> <span class="pl-c1">=</span> <span class="pl-s1">q_layer0</span>[<span class="pl-c1">0</span>]
<span class="pl-s1">q_layer0_head0</span>.<span class="pl-s1">shape</span></pre><div class="zeroclipboard-container">
    
  </div></div>
<div class="snippet-clipboard-content notranslate position-relative overflow-auto"><pre class="notranslate"><code>torch.Size([128, 4096])
</code></pre><div class="zeroclipboard-container">
    
  </div></div>
<div class="markdown-heading" dir="auto"><h3 tabindex="-1" class="heading-element" dir="auto" _msttexthash="186898660" _msthash="238">现在，我们将查询权重与 token 嵌入相乘，以接收对 token 的查询</h3><a id="user-content-we-now-multiply-the-query-weights-with-the-token-embedding-to-recive-a-query-for-the-token" class="anchor" aria-label="永久链接：我们现在将查询权重与 token 嵌入相乘，以接收对 token 的查询" href="#we-now-multiply-the-query-weights-with-the-token-embedding-to-recive-a-query-for-the-token" _mstaria-label="5802641" _msthash="239"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></div>
<p dir="auto" _msttexthash="518738181" _msthash="240">在这里，你可以看到结果的形状是 [17x128]，这是因为我们有 17 个标记，每个标记都有一个 128 长度的查询。</p>
<div dir="auto">
    <a target="_blank" rel="noopener noreferrer" href="https://github.com/naklecha/llama3-from-scratch/blob/main/images/q_per_token.png"><img src="https://github.com/naklecha/llama3-from-scratch/raw/main/images/q_per_token.png" width="600" style="max-width: 100%;"></a>
</div>
<div class="highlight highlight-source-python notranslate position-relative overflow-auto" dir="auto"><pre><span class="pl-s1">q_per_token</span> <span class="pl-c1">=</span> <span class="pl-s1">torch</span>.<span class="pl-en">matmul</span>(<span class="pl-s1">token_embeddings</span>, <span class="pl-s1">q_layer0_head0</span>.<span class="pl-v">T</span>)
<span class="pl-s1">q_per_token</span>.<span class="pl-s1">shape</span></pre><div class="zeroclipboard-container">
    <clipboard-copy aria-label="Copy" class="ClipboardButton btn btn-invisible js-clipboard-copy m-2 p-0 d-flex flex-justify-center flex-items-center" data-copy-feedback="Copied!" data-tooltip-direction="w" value="q_per_token = torch.matmul(token_embeddings, q_layer0_head0.T)
q_per_token.shape" tabindex="0" role="button">
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-copy js-clipboard-copy-icon">
    <path d="M0 6.75C0 5.784.784 5 1.75 5h1.5a.75.75 0 0 1 0 1.5h-1.5a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-1.5a.75.75 0 0 1 1.5 0v1.5A1.75 1.75 0 0 1 9.25 16h-7.5A1.75 1.75 0 0 1 0 14.25Z"></path><path d="M5 1.75C5 .784 5.784 0 6.75 0h7.5C15.216 0 16 .784 16 1.75v7.5A1.75 1.75 0 0 1 14.25 11h-7.5A1.75 1.75 0 0 1 5 9.25Zm1.75-.25a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-7.5a.25.25 0 0 0-.25-.25Z"></path>
</svg>
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-check js-clipboard-check-icon color-fg-success d-none">
    <path d="M13.78 4.22a.75.75 0 0 1 0 1.06l-7.25 7.25a.75.75 0 0 1-1.06 0L2.22 9.28a.751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018L6 10.94l6.72-6.72a.75.75 0 0 1 1.06 0Z"></path>
</svg>
    </clipboard-copy>
  </div></div>
<div class="snippet-clipboard-content notranslate position-relative overflow-auto"><pre class="notranslate"><code>torch.Size([17, 128])
</code></pre><div class="zeroclipboard-container">
    
  </div></div>
<div class="markdown-heading" dir="auto"><h2 tabindex="-1" class="heading-element" dir="auto" _msttexthash="12045462" _msthash="241">定位编码</h2><a id="user-content-positioning-encoding" class="anchor" aria-label="固定链接：定位编码" href="#positioning-encoding" _mstaria-label="833508" _msthash="242"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></div>
<p dir="auto"><font _mstmutation="1" _msttexthash="5317012402" _msthash="243">我们现在处于一个阶段，我们对于提示中的每个 token 都有一个查询向量，但如果你仔细想想 —— 单个查询向量不知道在提示中的位置。<br _mstmutation="1" _istranslated="1"><br _mstmutation="1" _istranslated="1"> query： “生命、宇宙和万物的终极问题的答案是 ”<br _mstmutation="1" _istranslated="1"><br _mstmutation="1" _istranslated="1">在我们的提示符中，我们使用了 “the” 三次，我们需要所有 3 个 “the” 标记的查询向量根据它们在查询中的位置具有不同的查询向量（每个大小为 [1x128]）。我们使用 RoPE （rotory positional embedding） 执行这些旋转。</font><br><br></p>
<div class="markdown-heading" dir="auto"><h3 tabindex="-1" class="heading-element" dir="auto" _msttexthash="2957409" _msthash="244">绳</h3><a id="user-content-rope" class="anchor" aria-label="永久链接：RoPE" href="#rope" _mstaria-label="228501" _msthash="245"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></div>
<p dir="auto" _msttexthash="128500047" _msthash="246">观看此视频（这是我观看的）以理解数学。<a href="https://www.youtube.com/watch?v=o29P0Kpobz0&amp;t=530s" rel="nofollow" _istranslated="1">https://www.youtube.com/watch?v=o29P0Kpobz0&amp;t=530s</a></p>
<div dir="auto">
    <a target="_blank" rel="noopener noreferrer" href="https://github.com/naklecha/llama3-from-scratch/blob/main/images/rope.png"><img src="https://github.com/naklecha/llama3-from-scratch/raw/main/images/rope.png" width="600" style="max-width: 100%;"></a>
</div>
<div class="highlight highlight-source-python notranslate position-relative overflow-auto" dir="auto"><pre><span class="pl-s1">q_per_token_split_into_pairs</span> <span class="pl-c1">=</span> <span class="pl-s1">q_per_token</span>.<span class="pl-en">float</span>().<span class="pl-en">view</span>(<span class="pl-s1">q_per_token</span>.<span class="pl-s1">shape</span>[<span class="pl-c1">0</span>], <span class="pl-c1">-</span><span class="pl-c1">1</span>, <span class="pl-c1">2</span>)
<span class="pl-s1">q_per_token_split_into_pairs</span>.<span class="pl-s1">shape</span></pre><div class="zeroclipboard-container">
    
  </div></div>
<div class="snippet-clipboard-content notranslate position-relative overflow-auto"><pre class="notranslate"><code>torch.Size([17, 64, 2])
</code></pre><div class="zeroclipboard-container">
    
  </div></div>
<p dir="auto" _msttexthash="2844560992" _msthash="247">在上面的步骤中，我们将查询向量分成几对，我们对每对应用旋转角度偏移！<br _istranslated="1"><br _istranslated="1">我们现在有一个大小为 [17x64x2] 的向量，这是 128 个长度的查询，对于提示中的每个标记，分为 64 对！这 64 对中的每一对都将由 m*（theta） 旋转，其中 m 是我们旋转查询的代币的位置！</p>
<div dir="auto">
    <a target="_blank" rel="noopener noreferrer" href="https://github.com/naklecha/llama3-from-scratch/blob/main/images/qsplit.png"><img src="https://github.com/naklecha/llama3-from-scratch/raw/main/images/qsplit.png" width="600" style="max-width: 100%;"></a>
</div>
<div class="markdown-heading" dir="auto"><h2 tabindex="-1" class="heading-element" dir="auto" _msttexthash="49757838" _msthash="248">使用复数的点积旋转向量</h2><a id="user-content-using-dot-product-of-complex-numbers-to-rotate-a-vector" class="anchor" aria-label="永久链接：使用复数的点积旋转向量" href="#using-dot-product-of-complex-numbers-to-rotate-a-vector" _mstaria-label="2723136" _msthash="249"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></div>
<div dir="auto">
    <a target="_blank" rel="noopener noreferrer" href="https://github.com/naklecha/llama3-from-scratch/blob/main/images/freq_cis.png"><img src="https://github.com/naklecha/llama3-from-scratch/raw/main/images/freq_cis.png" width="600" style="max-width: 100%;"></a>
</div>
<div class="highlight highlight-source-python notranslate position-relative overflow-auto" dir="auto"><pre><span class="pl-s1">zero_to_one_split_into_64_parts</span> <span class="pl-c1">=</span> <span class="pl-s1">torch</span>.<span class="pl-en">tensor</span>(<span class="pl-en">range</span>(<span class="pl-c1">64</span>))<span class="pl-c1">/</span><span class="pl-c1">64</span>
<span class="pl-s1">zero_to_one_split_into_64_parts</span></pre><div class="zeroclipboard-container">
     
  </div></div>
<div class="snippet-clipboard-content notranslate position-relative overflow-auto"><pre class="notranslate"><code>tensor([0.0000, 0.0156, 0.0312, 0.0469, 0.0625, 0.0781, 0.0938, 0.1094, 0.1250,
        0.1406, 0.1562, 0.1719, 0.1875, 0.2031, 0.2188, 0.2344, 0.2500, 0.2656,
        0.2812, 0.2969, 0.3125, 0.3281, 0.3438, 0.3594, 0.3750, 0.3906, 0.4062,
        0.4219, 0.4375, 0.4531, 0.4688, 0.4844, 0.5000, 0.5156, 0.5312, 0.5469,
        0.5625, 0.5781, 0.5938, 0.6094, 0.6250, 0.6406, 0.6562, 0.6719, 0.6875,
        0.7031, 0.7188, 0.7344, 0.7500, 0.7656, 0.7812, 0.7969, 0.8125, 0.8281,
        0.8438, 0.8594, 0.8750, 0.8906, 0.9062, 0.9219, 0.9375, 0.9531, 0.9688,
        0.9844])
</code></pre><div class="zeroclipboard-container">
     
  </div></div>
<div class="highlight highlight-source-python notranslate position-relative overflow-auto" dir="auto"><pre><span class="pl-s1">freqs</span> <span class="pl-c1">=</span> <span class="pl-c1">1.0</span> <span class="pl-c1">/</span> (<span class="pl-s1">rope_theta</span> <span class="pl-c1">**</span> <span class="pl-s1">zero_to_one_split_into_64_parts</span>)
<span class="pl-s1">freqs</span></pre><div class="zeroclipboard-container">
    
  </div></div>
<div class="snippet-clipboard-content notranslate position-relative overflow-auto"><pre class="notranslate"><code>tensor([1.0000e+00, 8.1462e-01, 6.6360e-01, 5.4058e-01, 4.4037e-01, 3.5873e-01,
        2.9223e-01, 2.3805e-01, 1.9392e-01, 1.5797e-01, 1.2869e-01, 1.0483e-01,
        8.5397e-02, 6.9566e-02, 5.6670e-02, 4.6164e-02, 3.7606e-02, 3.0635e-02,
        2.4955e-02, 2.0329e-02, 1.6560e-02, 1.3490e-02, 1.0990e-02, 8.9523e-03,
        7.2927e-03, 5.9407e-03, 4.8394e-03, 3.9423e-03, 3.2114e-03, 2.6161e-03,
        2.1311e-03, 1.7360e-03, 1.4142e-03, 1.1520e-03, 9.3847e-04, 7.6450e-04,
        6.2277e-04, 5.0732e-04, 4.1327e-04, 3.3666e-04, 2.7425e-04, 2.2341e-04,
        1.8199e-04, 1.4825e-04, 1.2077e-04, 9.8381e-05, 8.0143e-05, 6.5286e-05,
        5.3183e-05, 4.3324e-05, 3.5292e-05, 2.8750e-05, 2.3420e-05, 1.9078e-05,
        1.5542e-05, 1.2660e-05, 1.0313e-05, 8.4015e-06, 6.8440e-06, 5.5752e-06,
        4.5417e-06, 3.6997e-06, 3.0139e-06, 2.4551e-06])
</code></pre><div class="zeroclipboard-container">
     
  </div></div>
<div class="highlight highlight-source-python notranslate position-relative overflow-auto" dir="auto"><pre><span class="pl-s1">freqs_for_each_token</span> <span class="pl-c1">=</span> <span class="pl-s1">torch</span>.<span class="pl-en">outer</span>(<span class="pl-s1">torch</span>.<span class="pl-en">arange</span>(<span class="pl-c1">17</span>), <span class="pl-s1">freqs</span>)
<span class="pl-s1">freqs_cis</span> <span class="pl-c1">=</span> <span class="pl-s1">torch</span>.<span class="pl-en">polar</span>(<span class="pl-s1">torch</span>.<span class="pl-en">ones_like</span>(<span class="pl-s1">freqs_for_each_token</span>), <span class="pl-s1">freqs_for_each_token</span>)
<span class="pl-s1">freqs_cis</span>.<span class="pl-s1">shape</span>

<span class="pl-c"># viewing tjhe third row of freqs_cis</span>
<span class="pl-s1">value</span> <span class="pl-c1">=</span> <span class="pl-s1">freqs_cis</span>[<span class="pl-c1">3</span>]
<span class="pl-s1">plt</span>.<span class="pl-en">figure</span>()
<span class="pl-k">for</span> <span class="pl-s1">i</span>, <span class="pl-s1">element</span> <span class="pl-c1">in</span> <span class="pl-en">enumerate</span>(<span class="pl-s1">value</span>[:<span class="pl-c1">17</span>]):
    <span class="pl-s1">plt</span>.<span class="pl-en">plot</span>([<span class="pl-c1">0</span>, <span class="pl-s1">element</span>.<span class="pl-s1">real</span>], [<span class="pl-c1">0</span>, <span class="pl-s1">element</span>.<span class="pl-s1">imag</span>], <span class="pl-s1">color</span><span class="pl-c1">=</span><span class="pl-s">'blue'</span>, <span class="pl-s1">linewidth</span><span class="pl-c1">=</span><span class="pl-c1">1</span>, <span class="pl-s1">label</span><span class="pl-c1">=</span><span class="pl-s">f"Index: <span class="pl-s1"><span class="pl-kos">{</span><span class="pl-s1">i</span><span class="pl-kos">}</span></span>"</span>)
    <span class="pl-s1">plt</span>.<span class="pl-en">annotate</span>(<span class="pl-s">f"<span class="pl-s1"><span class="pl-kos">{</span><span class="pl-s1">i</span><span class="pl-kos">}</span></span>"</span>, <span class="pl-s1">xy</span><span class="pl-c1">=</span>(<span class="pl-s1">element</span>.<span class="pl-s1">real</span>, <span class="pl-s1">element</span>.<span class="pl-s1">imag</span>), <span class="pl-s1">color</span><span class="pl-c1">=</span><span class="pl-s">'red'</span>)
<span class="pl-s1">plt</span>.<span class="pl-en">xlabel</span>(<span class="pl-s">'Real'</span>)
<span class="pl-s1">plt</span>.<span class="pl-en">ylabel</span>(<span class="pl-s">'Imaginary'</span>)
<span class="pl-s1">plt</span>.<span class="pl-en">title</span>(<span class="pl-s">'Plot of one row of freqs_cis'</span>)
<span class="pl-s1">plt</span>.<span class="pl-en">show</span>()</pre><div class="zeroclipboard-container">
    
  </div></div>
<p dir="auto"><a target="_blank" rel="noopener noreferrer" href="https://github.com/naklecha/llama3-from-scratch/blob/main/images/implllama3_30_0.png"><img src="https://github.com/naklecha/llama3-from-scratch/raw/main/images/implllama3_30_0.png" alt="PNG 格式" style="max-width: 100%;" _mstalt="33683" _msthash="250"></a></p>
<div class="markdown-heading" dir="auto"><h3 tabindex="-1" class="heading-element" dir="auto" _msttexthash="265136586" _msthash="251">现在，我们为每个 token 的 query 元素提供了一个复数（角度变化向量）</h3><a id="user-content-now-that-we-have-a-complex-number-the-angle-change-vector-for-every-tokens-query-element" class="anchor" aria-label="永久链接：现在我们每个 token 的 query 元素都有一个复数（角度变化向量）" href="#now-that-we-have-a-complex-number-the-angle-change-vector-for-every-tokens-query-element" _mstaria-label="5810350" _msthash="252"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></div>
<p dir="auto" _msttexthash="500013631" _msthash="253">我们可以将查询（我们分成对的查询）转换为复数，然后 dot product 根据位置 <br _istranslated="1"> honeslty 旋转查询，这很好想想:)</p>
<div class="highlight highlight-source-python notranslate position-relative overflow-auto" dir="auto"><pre><span class="pl-s1">q_per_token_as_complex_numbers</span> <span class="pl-c1">=</span> <span class="pl-s1">torch</span>.<span class="pl-en">view_as_complex</span>(<span class="pl-s1">q_per_token_split_into_pairs</span>)
<span class="pl-s1">q_per_token_as_complex_numbers</span>.<span class="pl-s1">shape</span></pre><div class="zeroclipboard-container">
    <clipboard-copy aria-label="Copy" class="ClipboardButton btn btn-invisible js-clipboard-copy m-2 p-0 d-flex flex-justify-center flex-items-center" data-copy-feedback="Copied!" data-tooltip-direction="w" value="q_per_token_as_complex_numbers = torch.view_as_complex(q_per_token_split_into_pairs)
q_per_token_as_complex_numbers.shape" tabindex="0" role="button">
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-copy js-clipboard-copy-icon">
    <path d="M0 6.75C0 5.784.784 5 1.75 5h1.5a.75.75 0 0 1 0 1.5h-1.5a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-1.5a.75.75 0 0 1 1.5 0v1.5A1.75 1.75 0 0 1 9.25 16h-7.5A1.75 1.75 0 0 1 0 14.25Z"></path><path d="M5 1.75C5 .784 5.784 0 6.75 0h7.5C15.216 0 16 .784 16 1.75v7.5A1.75 1.75 0 0 1 14.25 11h-7.5A1.75 1.75 0 0 1 5 9.25Zm1.75-.25a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-7.5a.25.25 0 0 0-.25-.25Z"></path>
</svg>
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-check js-clipboard-check-icon color-fg-success d-none">
    <path d="M13.78 4.22a.75.75 0 0 1 0 1.06l-7.25 7.25a.75.75 0 0 1-1.06 0L2.22 9.28a.751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018L6 10.94l6.72-6.72a.75.75 0 0 1 1.06 0Z"></path>
</svg>
    </clipboard-copy>
  </div></div>
<div class="snippet-clipboard-content notranslate position-relative overflow-auto"><pre class="notranslate"><code>torch.Size([17, 64])
</code></pre><div class="zeroclipboard-container">
    
  </div></div>
<div class="highlight highlight-source-python notranslate position-relative overflow-auto" dir="auto"><pre><span class="pl-s1">q_per_token_as_complex_numbers_rotated</span> <span class="pl-c1">=</span> <span class="pl-s1">q_per_token_as_complex_numbers</span> <span class="pl-c1">*</span> <span class="pl-s1">freqs_cis</span>
<span class="pl-s1">q_per_token_as_complex_numbers_rotated</span>.<span class="pl-s1">shape</span></pre><div class="zeroclipboard-container">
    
  </div></div>
<div class="snippet-clipboard-content notranslate position-relative overflow-auto"><pre class="notranslate"><code>torch.Size([17, 64])
</code></pre><div class="zeroclipboard-container">
    
  </div></div>
<div class="markdown-heading" dir="auto"><h3 tabindex="-1" class="heading-element" dir="auto" _msttexthash="25659517" _msthash="254">获取旋转向量后</h3><a id="user-content-after-rotated-vector-is-obtained" class="anchor" aria-label="固定链接：获取旋转向量后" href="#after-rotated-vector-is-obtained" _mstaria-label="1334983" _msthash="255"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></div>
<p dir="auto" _msttexthash="141397061" _msthash="256">我们可以通过再次将复数视为实数来取回成对的查询</p>
<div class="highlight highlight-source-python notranslate position-relative overflow-auto" dir="auto"><pre><span class="pl-s1">q_per_token_split_into_pairs_rotated</span> <span class="pl-c1">=</span> <span class="pl-s1">torch</span>.<span class="pl-en">view_as_real</span>(<span class="pl-s1">q_per_token_as_complex_numbers_rotated</span>)
<span class="pl-s1">q_per_token_split_into_pairs_rotated</span>.<span class="pl-s1">shape</span></pre><div class="zeroclipboard-container">
     
  </div></div>
<div class="snippet-clipboard-content notranslate position-relative overflow-auto"><pre class="notranslate"><code>torch.Size([17, 64, 2])
</code></pre><div class="zeroclipboard-container">
    
  </div></div>
<p dir="auto" _msttexthash="767339560" _msthash="257">旋转对现在已合并，我们现在有一个形状为 [17x128] 的新查询向量（旋转查询向量），其中 17 是标记的数量，128 是查询向量的 dim</p>
<div class="highlight highlight-source-python notranslate position-relative overflow-auto" dir="auto"><pre><span class="pl-s1">q_per_token_rotated</span> <span class="pl-c1">=</span> <span class="pl-s1">q_per_token_split_into_pairs_rotated</span>.<span class="pl-en">view</span>(<span class="pl-s1">q_per_token</span>.<span class="pl-s1">shape</span>)
<span class="pl-s1">q_per_token_rotated</span>.<span class="pl-s1">shape</span></pre><div class="zeroclipboard-container">
    
  </div></div>
<div class="snippet-clipboard-content notranslate position-relative overflow-auto"><pre class="notranslate"><code>torch.Size([17, 128])
</code></pre><div class="zeroclipboard-container">
    
  </div></div>
<div class="markdown-heading" dir="auto"><h1 tabindex="-1" class="heading-element" dir="auto" _msttexthash="52995917" _msthash="258">键（与 queries 几乎相同）</h1><a id="user-content-keys-almost-the-same-as-queries" class="anchor" aria-label="永久链接：keys（与 queries 几乎相同）" href="#keys-almost-the-same-as-queries" _mstaria-label="1292759" _msthash="259"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></div>
<div dir="auto">
    <a target="_blank" rel="noopener noreferrer" href="https://github.com/naklecha/llama3-from-scratch/blob/main/images/keys.png"><img src="https://github.com/naklecha/llama3-from-scratch/raw/main/images/keys.png" width="600px" style="max-width: 100%;"></a>
</div><font _mstmutation="1" _msttexthash="2810206295" _msthash="260">我他妈的懒惰，所以我不打算对键进行数学运算，你唯一需要记住的是： <br _mstmutation="1" _istranslated="1"> &gt;键生成的键向量也是 128 个<br _mstmutation="1" _istranslated="1">&gt;键只有查询权重数量的 1/4，这是因为键的权重一次在 4 个头之间共享， 为了减少计算次数，需要<br _mstmutation="1" _istranslated="1">&gt;键也被轮换以添加位置信息，就像查询一样，因为同样的原因</font><div class="highlight highlight-source-python notranslate position-relative overflow-auto" dir="auto"><pre><span class="pl-s1">k_layer0</span> <span class="pl-c1">=</span> <span class="pl-s1">model</span>[<span class="pl-s">"layers.0.attention.wk.weight"</span>]
<span class="pl-s1">k_layer0</span> <span class="pl-c1">=</span> <span class="pl-s1">k_layer0</span>.<span class="pl-en">view</span>(<span class="pl-s1">n_kv_heads</span>, <span class="pl-s1">k_layer0</span>.<span class="pl-s1">shape</span>[<span class="pl-c1">0</span>] <span class="pl-c1">//</span> <span class="pl-s1">n_kv_heads</span>, <span class="pl-s1">dim</span>)
<span class="pl-s1">k_layer0</span>.<span class="pl-s1">shape</span></pre><div class="zeroclipboard-container">
    
  </div></div>
<div class="snippet-clipboard-content notranslate position-relative overflow-auto"><pre class="notranslate"><code>torch.Size([8, 128, 4096])
</code></pre><div class="zeroclipboard-container">
    
  </div></div>
<div class="highlight highlight-source-python notranslate position-relative overflow-auto" dir="auto"><pre><span class="pl-s1">k_layer0_head0</span> <span class="pl-c1">=</span> <span class="pl-s1">k_layer0</span>[<span class="pl-c1">0</span>]
<span class="pl-s1">k_layer0_head0</span>.<span class="pl-s1">shape</span></pre><div class="zeroclipboard-container">
    
  </div></div>
<div class="snippet-clipboard-content notranslate position-relative overflow-auto"><pre class="notranslate"><code>torch.Size([128, 4096])
</code></pre><div class="zeroclipboard-container">
    
  </div></div>
<div class="highlight highlight-source-python notranslate position-relative overflow-auto" dir="auto"><pre><span class="pl-s1">k_per_token</span> <span class="pl-c1">=</span> <span class="pl-s1">torch</span>.<span class="pl-en">matmul</span>(<span class="pl-s1">token_embeddings</span>, <span class="pl-s1">k_layer0_head0</span>.<span class="pl-v">T</span>)
<span class="pl-s1">k_per_token</span>.<span class="pl-s1">shape</span></pre><div class="zeroclipboard-container">
    
  </div></div>
<div class="snippet-clipboard-content notranslate position-relative overflow-auto"><pre class="notranslate"><code>torch.Size([17, 128])
</code></pre><div class="zeroclipboard-container">
    
  </div></div>
<div class="highlight highlight-source-python notranslate position-relative overflow-auto" dir="auto"><pre><span class="pl-s1">k_per_token_split_into_pairs</span> <span class="pl-c1">=</span> <span class="pl-s1">k_per_token</span>.<span class="pl-en">float</span>().<span class="pl-en">view</span>(<span class="pl-s1">k_per_token</span>.<span class="pl-s1">shape</span>[<span class="pl-c1">0</span>], <span class="pl-c1">-</span><span class="pl-c1">1</span>, <span class="pl-c1">2</span>)
<span class="pl-s1">k_per_token_split_into_pairs</span>.<span class="pl-s1">shape</span></pre><div class="zeroclipboard-container">
    
  </div></div>
<div class="snippet-clipboard-content notranslate position-relative overflow-auto"><pre class="notranslate"><code>torch.Size([17, 64, 2])
</code></pre><div class="zeroclipboard-container">
    
  </div></div>
<div class="highlight highlight-source-python notranslate position-relative overflow-auto" dir="auto"><pre><span class="pl-s1">k_per_token_as_complex_numbers</span> <span class="pl-c1">=</span> <span class="pl-s1">torch</span>.<span class="pl-en">view_as_complex</span>(<span class="pl-s1">k_per_token_split_into_pairs</span>)
<span class="pl-s1">k_per_token_as_complex_numbers</span>.<span class="pl-s1">shape</span></pre><div class="zeroclipboard-container">
    
  </div></div>
<div class="snippet-clipboard-content notranslate position-relative overflow-auto"><pre class="notranslate"><code>torch.Size([17, 64])
</code></pre><div class="zeroclipboard-container">
   
  </div></div>
<div class="highlight highlight-source-python notranslate position-relative overflow-auto" dir="auto"><pre><span class="pl-s1">k_per_token_split_into_pairs_rotated</span> <span class="pl-c1">=</span> <span class="pl-s1">torch</span>.<span class="pl-en">view_as_real</span>(<span class="pl-s1">k_per_token_as_complex_numbers</span> <span class="pl-c1">*</span> <span class="pl-s1">freqs_cis</span>)
<span class="pl-s1">k_per_token_split_into_pairs_rotated</span>.<span class="pl-s1">shape</span></pre><div class="zeroclipboard-container">
    
  </div></div>
<div class="snippet-clipboard-content notranslate position-relative overflow-auto"><pre class="notranslate"><code>torch.Size([17, 64, 2])
</code></pre><div class="zeroclipboard-container">
    
  </div></div>
<div class="highlight highlight-source-python notranslate position-relative overflow-auto" dir="auto"><pre><span class="pl-s1">k_per_token_rotated</span> <span class="pl-c1">=</span> <span class="pl-s1">k_per_token_split_into_pairs_rotated</span>.<span class="pl-en">view</span>(<span class="pl-s1">k_per_token</span>.<span class="pl-s1">shape</span>)
<span class="pl-s1">k_per_token_rotated</span>.<span class="pl-s1">shape</span></pre><div class="zeroclipboard-container">
    
  </div></div>
<div class="snippet-clipboard-content notranslate position-relative overflow-auto"><pre class="notranslate"><code>torch.Size([17, 128])
</code></pre><div class="zeroclipboard-container">
    
  </div></div>
<div class="markdown-heading" dir="auto"><h2 tabindex="-1" class="heading-element" dir="auto" _msttexthash="128206039" _msthash="261">在此阶段，现在每个令牌都有 queries 和 keys 的轮换值。</h2><a id="user-content-at-this-stage-now-have-both-the-rotated-values-of-queries-and-keys-for-each-token" class="anchor" aria-label="永久链接：在此阶段，现在每个令牌都有 queries 和 keys 的轮换值。" href="#at-this-stage-now-have-both-the-rotated-values-of-queries-and-keys-for-each-token" _mstaria-label="4820985" _msthash="262"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></div>
<div dir="auto">
    <a target="_blank" rel="noopener noreferrer" href="https://github.com/naklecha/llama3-from-scratch/blob/main/images/keys0.png"><img src="https://github.com/naklecha/llama3-from-scratch/raw/main/images/keys0.png" width="600px" style="max-width: 100%;"></a>
</div><font _mstmutation="1" _msttexthash="80051634" _msthash="263">现在，每个查询和键的形状都是 [17x128]。</font><div class="markdown-heading" dir="auto"><h2 tabindex="-1" class="heading-element" dir="auto" _msttexthash="95613700" _msthash="264">在下一步中，我们将乘以查询和键矩阵</h2><a id="user-content-in-the-next-step-we-will-multiply-the-queries-and-key-matrices" class="anchor" aria-label="永久链接：在下一步中，我们将乘以查询和键矩阵" href="#in-the-next-step-we-will-multiply-the-queries-and-key-matrices" _mstaria-label="3172819" _msthash="265"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></div>
<p dir="auto" _msttexthash="1633081697" _msthash="266">这样做会给我们一个分数，将每个 Token 相互映射<br _istranslated="1">，这个分数描述了每个 Token 的查询与每个 Token 的键的关联程度。
这就是 SELF ATTENTION :)<br _istranslated="1">注意力分数矩阵 （qk_per_token） 的形状为 [17x17]，其中 17 是提示中的标记数</p>
<div dir="auto">
    <a target="_blank" rel="noopener noreferrer" href="https://github.com/naklecha/llama3-from-scratch/blob/main/images/qkmatmul.png"><img src="https://github.com/naklecha/llama3-from-scratch/raw/main/images/qkmatmul.png" width="600px" style="max-width: 100%;"></a>
</div>
<div class="highlight highlight-source-python notranslate position-relative overflow-auto" dir="auto"><pre><span class="pl-s1">qk_per_token</span> <span class="pl-c1">=</span> <span class="pl-s1">torch</span>.<span class="pl-en">matmul</span>(<span class="pl-s1">q_per_token_rotated</span>, <span class="pl-s1">k_per_token_rotated</span>.<span class="pl-v">T</span>)<span class="pl-c1">/</span>(<span class="pl-s1">head_dim</span>)<span class="pl-c1">**</span><span class="pl-c1">0.5</span>
<span class="pl-s1">qk_per_token</span>.<span class="pl-s1">shape</span></pre><div class="zeroclipboard-container">
   
  </div></div>
<div class="snippet-clipboard-content notranslate position-relative overflow-auto"><pre class="notranslate"><code>torch.Size([17, 17])
</code></pre><div class="zeroclipboard-container">
   
  </div></div>
<div class="markdown-heading" dir="auto"><h1 tabindex="-1" class="heading-element" dir="auto" _msttexthash="30956588" _msthash="267">我们现在必须屏蔽 Query Key Scores</h1><a id="user-content-we-now-have-to-mask-query-key-scores" class="anchor" aria-label="永久链接：我们现在必须屏蔽查询键分数" href="#we-now-have-to-mask-query-key-scores" _mstaria-label="1430923" _msthash="268"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></div>
<p dir="auto" _msttexthash="1241779110" _msthash="269">在 llama3 的训练过程中，未来代币 QK 分数被屏蔽。<br _istranslated="1"> 为什么？因为在训练过程中，我们只学习使用过去的标记来预测标记。<br _istranslated="1">因此，在推理过程中，我们将 Future Tokens 设置为零。</p>
<div dir="auto">
    <a target="_blank" rel="noopener noreferrer" href="https://github.com/naklecha/llama3-from-scratch/blob/main/images/mask.png"><img src="https://github.com/naklecha/llama3-from-scratch/raw/main/images/mask.png" width="600px" style="max-width: 100%;"></a>
</div>
<div class="highlight highlight-source-python notranslate position-relative overflow-auto" dir="auto"><pre><span class="pl-k">def</span> <span class="pl-en">display_qk_heatmap</span>(<span class="pl-s1">qk_per_token</span>):
    <span class="pl-s1">_</span>, <span class="pl-s1">ax</span> <span class="pl-c1">=</span> <span class="pl-s1">plt</span>.<span class="pl-en">subplots</span>()
    <span class="pl-s1">im</span> <span class="pl-c1">=</span> <span class="pl-s1">ax</span>.<span class="pl-en">imshow</span>(<span class="pl-s1">qk_per_token</span>.<span class="pl-en">to</span>(<span class="pl-s1">float</span>).<span class="pl-en">detach</span>(), <span class="pl-s1">cmap</span><span class="pl-c1">=</span><span class="pl-s">'viridis'</span>)
    <span class="pl-s1">ax</span>.<span class="pl-en">set_xticks</span>(<span class="pl-en">range</span>(<span class="pl-en">len</span>(<span class="pl-s1">prompt_split_as_tokens</span>)))
    <span class="pl-s1">ax</span>.<span class="pl-en">set_yticks</span>(<span class="pl-en">range</span>(<span class="pl-en">len</span>(<span class="pl-s1">prompt_split_as_tokens</span>)))
    <span class="pl-s1">ax</span>.<span class="pl-en">set_xticklabels</span>(<span class="pl-s1">prompt_split_as_tokens</span>)
    <span class="pl-s1">ax</span>.<span class="pl-en">set_yticklabels</span>(<span class="pl-s1">prompt_split_as_tokens</span>)
    <span class="pl-s1">ax</span>.<span class="pl-s1">figure</span>.<span class="pl-en">colorbar</span>(<span class="pl-s1">im</span>, <span class="pl-s1">ax</span><span class="pl-c1">=</span><span class="pl-s1">ax</span>)
    
<span class="pl-en">display_qk_heatmap</span>(<span class="pl-s1">qk_per_token</span>)</pre><div class="zeroclipboard-container">
    
  </div></div>
<p dir="auto"><a target="_blank" rel="noopener noreferrer" href="https://github.com/naklecha/llama3-from-scratch/blob/main/images/implllama3_50_0.png"><img src="https://github.com/naklecha/llama3-from-scratch/raw/main/images/implllama3_50_0.png" alt="PNG 格式" style="max-width: 100%;" _mstalt="33683" _msthash="270"></a></p>
<div class="highlight highlight-source-python notranslate position-relative overflow-auto" dir="auto"><pre><span class="pl-s1">mask</span> <span class="pl-c1">=</span> <span class="pl-s1">torch</span>.<span class="pl-en">full</span>((<span class="pl-en">len</span>(<span class="pl-s1">tokens</span>), <span class="pl-en">len</span>(<span class="pl-s1">tokens</span>)), <span class="pl-en">float</span>(<span class="pl-s">"-inf"</span>), <span class="pl-s1">device</span><span class="pl-c1">=</span><span class="pl-s1">tokens</span>.<span class="pl-s1">device</span>)
<span class="pl-s1">mask</span> <span class="pl-c1">=</span> <span class="pl-s1">torch</span>.<span class="pl-en">triu</span>(<span class="pl-s1">mask</span>, <span class="pl-s1">diagonal</span><span class="pl-c1">=</span><span class="pl-c1">1</span>)
<span class="pl-s1">mask</span></pre><div class="zeroclipboard-container">
    
  </div></div>
<div class="snippet-clipboard-content notranslate position-relative overflow-auto"><pre class="notranslate"><code>tensor([[0., -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf],
        [0., 0., -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf],
        [0., 0., 0., -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf],
        [0., 0., 0., 0., -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf],
        [0., 0., 0., 0., 0., -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf],
        [0., 0., 0., 0., 0., 0., -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf],
        [0., 0., 0., 0., 0., 0., 0., -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf],
        [0., 0., 0., 0., 0., 0., 0., 0., -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., -inf, -inf, -inf, -inf, -inf, -inf, -inf],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., -inf, -inf, -inf, -inf, -inf, -inf],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., -inf, -inf, -inf, -inf, -inf],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., -inf, -inf, -inf, -inf],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., -inf, -inf, -inf],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., -inf, -inf],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., -inf],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
</code></pre><div class="zeroclipboard-container">
    
  </div></div>
<div class="highlight highlight-source-python notranslate position-relative overflow-auto" dir="auto"><pre><span class="pl-s1">qk_per_token_after_masking</span> <span class="pl-c1">=</span> <span class="pl-s1">qk_per_token</span> <span class="pl-c1">+</span> <span class="pl-s1">mask</span>
<span class="pl-en">display_qk_heatmap</span>(<span class="pl-s1">qk_per_token_after_masking</span>)</pre><div class="zeroclipboard-container">
    
  </div></div>
<p dir="auto"><a target="_blank" rel="noopener noreferrer" href="https://github.com/naklecha/llama3-from-scratch/blob/main/images/implllama3_52_0.png"><img src="https://github.com/naklecha/llama3-from-scratch/raw/main/images/implllama3_52_0.png" alt="PNG 格式" style="max-width: 100%;" _mstalt="33683" _msthash="271"></a></p>
<div dir="auto">
    <a target="_blank" rel="noopener noreferrer" href="https://github.com/naklecha/llama3-from-scratch/blob/main/images/softmax.png"><img src="https://github.com/naklecha/llama3-from-scratch/raw/main/images/softmax.png" width="600px" style="max-width: 100%;"></a>
</div>
<div class="highlight highlight-source-python notranslate position-relative overflow-auto" dir="auto"><pre><span class="pl-s1">qk_per_token_after_masking_after_softmax</span> <span class="pl-c1">=</span> <span class="pl-s1">torch</span>.<span class="pl-s1">nn</span>.<span class="pl-s1">functional</span>.<span class="pl-en">softmax</span>(<span class="pl-s1">qk_per_token_after_masking</span>, <span class="pl-s1">dim</span><span class="pl-c1">=</span><span class="pl-c1">1</span>).<span class="pl-en">to</span>(<span class="pl-s1">torch</span>.<span class="pl-s1">bfloat16</span>)
<span class="pl-en">display_qk_heatmap</span>(<span class="pl-s1">qk_per_token_after_masking_after_softmax</span>)</pre><div class="zeroclipboard-container">
     
  </div></div>
<p dir="auto"><a target="_blank" rel="noopener noreferrer" href="https://github.com/naklecha/llama3-from-scratch/blob/main/images/implllama3_54_0.png"><img src="https://github.com/naklecha/llama3-from-scratch/raw/main/images/implllama3_54_0.png" alt="PNG 格式" style="max-width: 100%;" _mstalt="33683" _msthash="272"></a></p>
<div class="markdown-heading" dir="auto"><h2 tabindex="-1" class="heading-element" dir="auto" _msttexthash="64177477" _msthash="273">值（几乎是注意力的终点）</h2><a id="user-content-values-almost-the-end-of-attention" class="anchor" aria-label="Permalink： values （几乎是注意力的终点）" href="#values-almost-the-end-of-attention" _mstaria-label="1480999" _msthash="274"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></div>
<div dir="auto">
    <a target="_blank" rel="noopener noreferrer" href="https://github.com/naklecha/llama3-from-scratch/blob/main/images/value.png"><img src="https://github.com/naklecha/llama3-from-scratch/raw/main/images/value.png" width="600px" style="max-width: 100%;"></a>
</div><font _mstmutation="1" _msttexthash="1129190218" _msthash="275">这些分数 （0-1） 用于确定每个 Token <br _mstmutation="1" _istranslated="1"> 使用了多少值矩阵&gt;就像键一样，值权重也是每 4 个注意力头共享的（以节省计算） <br _mstmutation="1" _istranslated="1"> &gt;因此，下面的值权重矩阵的形状是 [8x128x4096]</font><div class="highlight highlight-source-python notranslate position-relative overflow-auto" dir="auto"><pre><span class="pl-s1">v_layer0</span> <span class="pl-c1">=</span> <span class="pl-s1">model</span>[<span class="pl-s">"layers.0.attention.wv.weight"</span>]
<span class="pl-s1">v_layer0</span> <span class="pl-c1">=</span> <span class="pl-s1">v_layer0</span>.<span class="pl-en">view</span>(<span class="pl-s1">n_kv_heads</span>, <span class="pl-s1">v_layer0</span>.<span class="pl-s1">shape</span>[<span class="pl-c1">0</span>] <span class="pl-c1">//</span> <span class="pl-s1">n_kv_heads</span>, <span class="pl-s1">dim</span>)
<span class="pl-s1">v_layer0</span>.<span class="pl-s1">shape</span></pre><div class="zeroclipboard-container">
    
  </div></div>
<div class="snippet-clipboard-content notranslate position-relative overflow-auto"><pre class="notranslate"><code>torch.Size([8, 128, 4096])
</code></pre><div class="zeroclipboard-container">
     
  </div></div>
<p dir="auto" _msttexthash="85262775" _msthash="276">第一层，第一个 head value 权重矩阵如下</p>
<div class="highlight highlight-source-python notranslate position-relative overflow-auto" dir="auto"><pre><span class="pl-s1">v_layer0_head0</span> <span class="pl-c1">=</span> <span class="pl-s1">v_layer0</span>[<span class="pl-c1">0</span>]
<span class="pl-s1">v_layer0_head0</span>.<span class="pl-s1">shape</span></pre><div class="zeroclipboard-container">
    
  </div></div>
<div class="snippet-clipboard-content notranslate position-relative overflow-auto"><pre class="notranslate"><code>torch.Size([128, 4096])
</code></pre><div class="zeroclipboard-container">
    
  </div></div>
<div class="markdown-heading" dir="auto"><h2 tabindex="-1" class="heading-element" dir="auto" _msttexthash="8474583" _msthash="277">值向量</h2><a id="user-content-value-vectors" class="anchor" aria-label="永久链接：值向量" href="#value-vectors" _mstaria-label="532077" _msthash="278"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></div>
<div dir="auto">
    <a target="_blank" rel="noopener noreferrer" href="https://github.com/naklecha/llama3-from-scratch/blob/main/images/v0.png"><img src="https://github.com/naklecha/llama3-from-scratch/raw/main/images/v0.png" width="600px" style="max-width: 100%;"></a>
</div><font _mstmutation="1" _msttexthash="789544223" _msthash="279">我们现在使用 Value Weghts 来获取每个标记的 attention 值，其大小为 [17x128]，其中 17 是提示中的标记数，128 是每个标记的值向量的 dim</font><div class="highlight highlight-source-python notranslate position-relative overflow-auto" dir="auto"><pre><span class="pl-s1">v_per_token</span> <span class="pl-c1">=</span> <span class="pl-s1">torch</span>.<span class="pl-en">matmul</span>(<span class="pl-s1">token_embeddings</span>, <span class="pl-s1">v_layer0_head0</span>.<span class="pl-v">T</span>)
<span class="pl-s1">v_per_token</span>.<span class="pl-s1">shape</span></pre><div class="zeroclipboard-container">
   
  </div></div>
<div class="snippet-clipboard-content notranslate position-relative overflow-auto"><pre class="notranslate"><code>torch.Size([17, 128])
</code></pre><div class="zeroclipboard-container">
    
  </div></div>
<div class="markdown-heading" dir="auto"><h2 tabindex="-1" class="heading-element" dir="auto" _msttexthash="7595367" _msthash="280">注意力</h2><a id="user-content-attention" class="anchor" aria-label="永久链接：注意" href="#attention" _mstaria-label="415181" _msthash="281"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></div>
<div dir="auto">
    <a target="_blank" rel="noopener noreferrer" href="https://github.com/naklecha/llama3-from-scratch/blob/main/images/attention.png"><img src="https://github.com/naklecha/llama3-from-scratch/raw/main/images/attention.png" width="600px" style="max-width: 100%;"></a>
</div><font _mstmutation="1" _msttexthash="150436546" _msthash="282">与每个标记的值相乘后，生成的注意力向量的形状为 [17*128]</font><div class="highlight highlight-source-python notranslate position-relative overflow-auto" dir="auto"><pre><span class="pl-s1">qkv_attention</span> <span class="pl-c1">=</span> <span class="pl-s1">torch</span>.<span class="pl-en">matmul</span>(<span class="pl-s1">qk_per_token_after_masking_after_softmax</span>, <span class="pl-s1">v_per_token</span>)
<span class="pl-s1">qkv_attention</span>.<span class="pl-s1">shape</span></pre><div class="zeroclipboard-container">
    
  </div></div>
<div class="snippet-clipboard-content notranslate position-relative overflow-auto"><pre class="notranslate"><code>torch.Size([17, 128])
</code></pre><div class="zeroclipboard-container">
    
  </div></div>
<div class="markdown-heading" dir="auto"><h1 tabindex="-1" class="heading-element" dir="auto" _msttexthash="13966745" _msthash="283">多头注意力</h1><a id="user-content-multi-head-attention" class="anchor" aria-label="永久链接：多头关注" href="#multi-head-attention" _mstaria-label="789750" _msthash="284"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></div>
<div dir="auto">
    <a target="_blank" rel="noopener noreferrer" href="https://github.com/naklecha/llama3-from-scratch/blob/main/images/heads.png"><img src="https://github.com/naklecha/llama3-from-scratch/raw/main/images/heads.png" width="600px" style="max-width: 100%;"></a>
</div><font _mstmutation="1" _msttexthash="2721902248" _msthash="285">我们现在有了第一层和第一个头的注意力值现在<br _mstmutation="1" _istranslated="1">我要运行一个循环并执行与上面的单元格完全相同的数学运算，但对于第一层中的每个头，我们现在有一个第一层上所有 32 个头的 qkv_attention 矩阵，接下来我要把所有注意力分数合并成一个大小为 [17x4096] <br _mstmutation="1" _istranslated="1"> 的大矩阵，我们快到了最后:)</font><div class="highlight highlight-source-python notranslate position-relative overflow-auto" dir="auto"><pre><span class="pl-s1">qkv_attention_store</span> <span class="pl-c1">=</span> []

<span class="pl-k">for</span> <span class="pl-s1">head</span> <span class="pl-c1">in</span> <span class="pl-en">range</span>(<span class="pl-s1">n_heads</span>):
    <span class="pl-s1">q_layer0_head</span> <span class="pl-c1">=</span> <span class="pl-s1">q_layer0</span>[<span class="pl-s1">head</span>]
    <span class="pl-s1">k_layer0_head</span> <span class="pl-c1">=</span> <span class="pl-s1">k_layer0</span>[<span class="pl-s1">head</span><span class="pl-c1">//</span><span class="pl-c1">4</span>] <span class="pl-c"># key weights are shared across 4 heads</span>
    <span class="pl-s1">v_layer0_head</span> <span class="pl-c1">=</span> <span class="pl-s1">v_layer0</span>[<span class="pl-s1">head</span><span class="pl-c1">//</span><span class="pl-c1">4</span>] <span class="pl-c"># value weights are shared across 4 heads</span>
    <span class="pl-s1">q_per_token</span> <span class="pl-c1">=</span> <span class="pl-s1">torch</span>.<span class="pl-en">matmul</span>(<span class="pl-s1">token_embeddings</span>, <span class="pl-s1">q_layer0_head</span>.<span class="pl-v">T</span>)
    <span class="pl-s1">k_per_token</span> <span class="pl-c1">=</span> <span class="pl-s1">torch</span>.<span class="pl-en">matmul</span>(<span class="pl-s1">token_embeddings</span>, <span class="pl-s1">k_layer0_head</span>.<span class="pl-v">T</span>)
    <span class="pl-s1">v_per_token</span> <span class="pl-c1">=</span> <span class="pl-s1">torch</span>.<span class="pl-en">matmul</span>(<span class="pl-s1">token_embeddings</span>, <span class="pl-s1">v_layer0_head</span>.<span class="pl-v">T</span>)

    <span class="pl-s1">q_per_token_split_into_pairs</span> <span class="pl-c1">=</span> <span class="pl-s1">q_per_token</span>.<span class="pl-en">float</span>().<span class="pl-en">view</span>(<span class="pl-s1">q_per_token</span>.<span class="pl-s1">shape</span>[<span class="pl-c1">0</span>], <span class="pl-c1">-</span><span class="pl-c1">1</span>, <span class="pl-c1">2</span>)
    <span class="pl-s1">q_per_token_as_complex_numbers</span> <span class="pl-c1">=</span> <span class="pl-s1">torch</span>.<span class="pl-en">view_as_complex</span>(<span class="pl-s1">q_per_token_split_into_pairs</span>)
    <span class="pl-s1">q_per_token_split_into_pairs_rotated</span> <span class="pl-c1">=</span> <span class="pl-s1">torch</span>.<span class="pl-en">view_as_real</span>(<span class="pl-s1">q_per_token_as_complex_numbers</span> <span class="pl-c1">*</span> <span class="pl-s1">freqs_cis</span>[:<span class="pl-en">len</span>(<span class="pl-s1">tokens</span>)])
    <span class="pl-s1">q_per_token_rotated</span> <span class="pl-c1">=</span> <span class="pl-s1">q_per_token_split_into_pairs_rotated</span>.<span class="pl-en">view</span>(<span class="pl-s1">q_per_token</span>.<span class="pl-s1">shape</span>)

    <span class="pl-s1">k_per_token_split_into_pairs</span> <span class="pl-c1">=</span> <span class="pl-s1">k_per_token</span>.<span class="pl-en">float</span>().<span class="pl-en">view</span>(<span class="pl-s1">k_per_token</span>.<span class="pl-s1">shape</span>[<span class="pl-c1">0</span>], <span class="pl-c1">-</span><span class="pl-c1">1</span>, <span class="pl-c1">2</span>)
    <span class="pl-s1">k_per_token_as_complex_numbers</span> <span class="pl-c1">=</span> <span class="pl-s1">torch</span>.<span class="pl-en">view_as_complex</span>(<span class="pl-s1">k_per_token_split_into_pairs</span>)
    <span class="pl-s1">k_per_token_split_into_pairs_rotated</span> <span class="pl-c1">=</span> <span class="pl-s1">torch</span>.<span class="pl-en">view_as_real</span>(<span class="pl-s1">k_per_token_as_complex_numbers</span> <span class="pl-c1">*</span> <span class="pl-s1">freqs_cis</span>[:<span class="pl-en">len</span>(<span class="pl-s1">tokens</span>)])
    <span class="pl-s1">k_per_token_rotated</span> <span class="pl-c1">=</span> <span class="pl-s1">k_per_token_split_into_pairs_rotated</span>.<span class="pl-en">view</span>(<span class="pl-s1">k_per_token</span>.<span class="pl-s1">shape</span>)

    <span class="pl-s1">qk_per_token</span> <span class="pl-c1">=</span> <span class="pl-s1">torch</span>.<span class="pl-en">matmul</span>(<span class="pl-s1">q_per_token_rotated</span>, <span class="pl-s1">k_per_token_rotated</span>.<span class="pl-v">T</span>)<span class="pl-c1">/</span>(<span class="pl-c1">128</span>)<span class="pl-c1">**</span><span class="pl-c1">0.5</span>
    <span class="pl-s1">mask</span> <span class="pl-c1">=</span> <span class="pl-s1">torch</span>.<span class="pl-en">full</span>((<span class="pl-en">len</span>(<span class="pl-s1">tokens</span>), <span class="pl-en">len</span>(<span class="pl-s1">tokens</span>)), <span class="pl-en">float</span>(<span class="pl-s">"-inf"</span>), <span class="pl-s1">device</span><span class="pl-c1">=</span><span class="pl-s1">tokens</span>.<span class="pl-s1">device</span>)
    <span class="pl-s1">mask</span> <span class="pl-c1">=</span> <span class="pl-s1">torch</span>.<span class="pl-en">triu</span>(<span class="pl-s1">mask</span>, <span class="pl-s1">diagonal</span><span class="pl-c1">=</span><span class="pl-c1">1</span>)
    <span class="pl-s1">qk_per_token_after_masking</span> <span class="pl-c1">=</span> <span class="pl-s1">qk_per_token</span> <span class="pl-c1">+</span> <span class="pl-s1">mask</span>
    <span class="pl-s1">qk_per_token_after_masking_after_softmax</span> <span class="pl-c1">=</span> <span class="pl-s1">torch</span>.<span class="pl-s1">nn</span>.<span class="pl-s1">functional</span>.<span class="pl-en">softmax</span>(<span class="pl-s1">qk_per_token_after_masking</span>, <span class="pl-s1">dim</span><span class="pl-c1">=</span><span class="pl-c1">1</span>).<span class="pl-en">to</span>(<span class="pl-s1">torch</span>.<span class="pl-s1">bfloat16</span>)
    <span class="pl-s1">qkv_attention</span> <span class="pl-c1">=</span> <span class="pl-s1">torch</span>.<span class="pl-en">matmul</span>(<span class="pl-s1">qk_per_token_after_masking_after_softmax</span>, <span class="pl-s1">v_per_token</span>)
    <span class="pl-s1">qkv_attention</span> <span class="pl-c1">=</span> <span class="pl-s1">torch</span>.<span class="pl-en">matmul</span>(<span class="pl-s1">qk_per_token_after_masking_after_softmax</span>, <span class="pl-s1">v_per_token</span>)
    <span class="pl-s1">qkv_attention_store</span>.<span class="pl-en">append</span>(<span class="pl-s1">qkv_attention</span>)

<span class="pl-en">len</span>(<span class="pl-s1">qkv_attention_store</span>)</pre><div class="zeroclipboard-container">
    
  </div></div><div class="snippet-clipboard-content notranslate position-relative overflow-auto"><pre class="notranslate"><code>32
</code></pre><div class="zeroclipboard-container">
    <clipboard-copy aria-label="Copy" class="ClipboardButton btn btn-invisible js-clipboard-copy m-2 p-0 d-flex flex-justify-center flex-items-center" data-copy-feedback="Copied!" data-tooltip-direction="w" value="32" tabindex="0" role="button">
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-copy js-clipboard-copy-icon">
    <path d="M0 6.75C0 5.784.784 5 1.75 5h1.5a.75.75 0 0 1 0 1.5h-1.5a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-1.5a.75.75 0 0 1 1.5 0v1.5A1.75 1.75 0 0 1 9.25 16h-7.5A1.75 1.75 0 0 1 0 14.25Z"></path><path d="M5 1.75C5 .784 5.784 0 6.75 0h7.5C15.216 0 16 .784 16 1.75v7.5A1.75 1.75 0 0 1 14.25 11h-7.5A1.75 1.75 0 0 1 5 9.25Zm1.75-.25a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-7.5a.25.25 0 0 0-.25-.25Z"></path>
</svg>
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-check js-clipboard-check-icon color-fg-success d-none">
    <path d="M13.78 4.22a.75.75 0 0 1 0 1.06l-7.25 7.25a.75.75 0 0 1-1.06 0L2.22 9.28a.751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018L6 10.94l6.72-6.72a.75.75 0 0 1 1.06 0Z"></path>
</svg>
    </clipboard-copy>
  </div></div><div dir="auto">
    <a target="_blank" rel="noopener noreferrer" href="https://github.com/naklecha/llama3-from-scratch/blob/main/images/stacked.png"><img src="https://github.com/naklecha/llama3-from-scratch/raw/main/images/stacked.png" width="600px" style="max-width: 100%;"></a>
</div><div class="highlight highlight-source-python notranslate position-relative overflow-auto" dir="auto"><pre><span class="pl-s1">stacked_qkv_attention</span> <span class="pl-c1">=</span> <span class="pl-s1">torch</span>.<span class="pl-en">cat</span>(<span class="pl-s1">qkv_attention_store</span>, <span class="pl-s1">dim</span><span class="pl-c1">=</span><span class="pl-c1">-</span><span class="pl-c1">1</span>)
<span class="pl-s1">stacked_qkv_attention</span>.<span class="pl-s1">shape</span></pre><div class="zeroclipboard-container">
   
  </div></div>
<div class="snippet-clipboard-content notranslate position-relative overflow-auto"><pre class="notranslate"><code>torch.Size([17, 4096])
</code></pre><div class="zeroclipboard-container">
    
  </div></div>
<div class="markdown-heading" dir="auto"><h1 tabindex="-1" class="heading-element" dir="auto" _msttexthash="71574828" _msthash="286">Weight Matrix，最后的步骤之一</h1><a id="user-content-weight-matrix-one-of-the-final-steps" class="anchor" aria-label="永久链接：权重矩阵，最后的步骤之一" href="#weight-matrix-one-of-the-final-steps" _mstaria-label="1517958" _msthash="287"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></div>
<div dir="auto">
    <a target="_blank" rel="noopener noreferrer" href="https://github.com/naklecha/llama3-from-scratch/blob/main/images/weightmatrix.png"><img src="https://github.com/naklecha/llama3-from-scratch/raw/main/images/weightmatrix.png" width="600px" style="max-width: 100%;"></a>
</div><font _mstmutation="1" _msttexthash="166703199" _msthash="288">对于第 0 层 Attention 来说，最后要做的一件事是，将</font><div class="highlight highlight-source-python notranslate position-relative overflow-auto" dir="auto"><pre><span class="pl-s1">w_layer0</span> <span class="pl-c1">=</span> <span class="pl-s1">model</span>[<span class="pl-s">"layers.0.attention.wo.weight"</span>]
<span class="pl-s1">w_layer0</span>.<span class="pl-s1">shape</span></pre><div class="zeroclipboard-container">
    
  </div></div>
<div class="snippet-clipboard-content notranslate position-relative overflow-auto"><pre class="notranslate"><code>torch.Size([4096, 4096])
</code></pre><div class="zeroclipboard-container">
    
  </div></div>
<div class="markdown-heading" dir="auto"><h3 tabindex="-1" class="heading-element" dir="auto" _msttexthash="94878537" _msthash="289">这是一个简单的线性层，所以我们只需 matmul</h3><a id="user-content-this-is-a-simple-linear-layer-so-we-just-matmul" class="anchor" aria-label="永久链接：这是一个简单的线性层，所以我们只需 matmul" href="#this-is-a-simple-linear-layer-so-we-just-matmul" _mstaria-label="2106351" _msthash="290"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></div>
<div class="highlight highlight-source-python notranslate position-relative overflow-auto" dir="auto"><pre><span class="pl-s1">embedding_delta</span> <span class="pl-c1">=</span> <span class="pl-s1">torch</span>.<span class="pl-en">matmul</span>(<span class="pl-s1">stacked_qkv_attention</span>, <span class="pl-s1">w_layer0</span>.<span class="pl-v">T</span>)
<span class="pl-s1">embedding_delta</span>.<span class="pl-s1">shape</span></pre><div class="zeroclipboard-container">
    
  </div></div>
<div class="snippet-clipboard-content notranslate position-relative overflow-auto"><pre class="notranslate"><code>torch.Size([17, 4096])
</code></pre><div class="zeroclipboard-container">
     
  </div></div>
<div dir="auto">
    <a target="_blank" rel="noopener noreferrer" href="/naklecha/llama3-from-scratch/blob/main/images/afterattention.png"><img src="/naklecha/llama3-from-scratch/raw/main/images/afterattention.png" width="600px" style="max-width: 100%;"></a>
</div><font _mstmutation="1" _msttexthash="282259250" _msthash="291">现在，我们在 attention 之后有了 embedding 值的变化，这应该添加到原始的 token embedding 中</font><div class="highlight highlight-source-python notranslate position-relative overflow-auto" dir="auto"><pre><span class="pl-s1">embedding_after_edit</span> <span class="pl-c1">=</span> <span class="pl-s1">token_embeddings_unnormalized</span> <span class="pl-c1">+</span> <span class="pl-s1">embedding_delta</span>
<span class="pl-s1">embedding_after_edit</span>.<span class="pl-s1">shape</span></pre><div class="zeroclipboard-container">
    
  </div></div>
<div class="snippet-clipboard-content notranslate position-relative overflow-auto"><pre class="notranslate"><code>torch.Size([17, 4096])
</code></pre><div class="zeroclipboard-container">
    
  </div></div>
<div class="markdown-heading" dir="auto"><h2 tabindex="-1" class="heading-element" dir="auto" _msttexthash="155114141" _msthash="292">我们归一化，然后通过嵌入增量运行前馈神经网络</h2><a id="user-content-we-normalize-and-then-run-a-feed-forward-neural-network-through-the-embedding-delta" class="anchor" aria-label="永久链接：我们归一化，然后通过嵌入 delta 运行前馈神经网络" href="#we-normalize-and-then-run-a-feed-forward-neural-network-through-the-embedding-delta" _mstaria-label="5150912" _msthash="293"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></div>
<div dir="auto">
    <a target="_blank" rel="noopener noreferrer" href="/naklecha/llama3-from-scratch/blob/main/images/norm_after.png"><img src="/naklecha/llama3-from-scratch/raw/main/images/norm_after.png" width="600px" style="max-width: 100%;"></a>
</div>
<div class="highlight highlight-source-python notranslate position-relative overflow-auto" dir="auto"><pre><span class="pl-s1">embedding_after_edit_normalized</span> <span class="pl-c1">=</span> <span class="pl-en">rms_norm</span>(<span class="pl-s1">embedding_after_edit</span>, <span class="pl-s1">model</span>[<span class="pl-s">"layers.0.ffn_norm.weight"</span>])
<span class="pl-s1">embedding_after_edit_normalized</span>.<span class="pl-s1">shape</span></pre><div class="zeroclipboard-container">
     
  </div></div>
<div class="snippet-clipboard-content notranslate position-relative overflow-auto"><pre class="notranslate"><code>torch.Size([17, 4096])
</code></pre><div class="zeroclipboard-container">
    
  </div></div>
<div class="markdown-heading" dir="auto"><h2 tabindex="-1" class="heading-element" dir="auto" _msttexthash="58212635" _msthash="294">加载 FF 权重并实现前馈网络</h2><a id="user-content-loading-the-ff-weights-and-implementing-the-feed-forward-network" class="anchor" aria-label="永久链接：加载 ff 权重并实现前馈网络" href="#loading-the-ff-weights-and-implementing-the-feed-forward-network" _mstaria-label="3521726" _msthash="295"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></div>
<div dir="auto">
    <a target="_blank" rel="noopener noreferrer" href="https://github.com/naklecha/llama3-from-scratch/blob/main/images/swiglu.png"><img src="https://github.com/naklecha/llama3-from-scratch/raw/main/images/swiglu.png" width="600px" style="max-width: 100%;"></a>
</div><font _mstmutation="1" _msttexthash="1003838576" _msthash="296">在 llama3 中，他们使用了 SwiGLU 前馈网络，这种网络架构非常擅长在模型需要时添加非线性。<br _mstmutation="1" _istranslated="1">如今，在 LLMS 中使用这种前馈网络架构是非常标准的</font><div class="highlight highlight-source-python notranslate position-relative overflow-auto" dir="auto"><pre><span class="pl-s1">w1</span> <span class="pl-c1">=</span> <span class="pl-s1">model</span>[<span class="pl-s">"layers.0.feed_forward.w1.weight"</span>]
<span class="pl-s1">w2</span> <span class="pl-c1">=</span> <span class="pl-s1">model</span>[<span class="pl-s">"layers.0.feed_forward.w2.weight"</span>]
<span class="pl-s1">w3</span> <span class="pl-c1">=</span> <span class="pl-s1">model</span>[<span class="pl-s">"layers.0.feed_forward.w3.weight"</span>]
<span class="pl-s1">output_after_feedforward</span> <span class="pl-c1">=</span> <span class="pl-s1">torch</span>.<span class="pl-en">matmul</span>(<span class="pl-s1">torch</span>.<span class="pl-s1">functional</span>.<span class="pl-v">F</span>.<span class="pl-en">silu</span>(<span class="pl-s1">torch</span>.<span class="pl-en">matmul</span>(<span class="pl-s1">embedding_after_edit_normalized</span>, <span class="pl-s1">w1</span>.<span class="pl-v">T</span>)) <span class="pl-c1">*</span> <span class="pl-s1">torch</span>.<span class="pl-en">matmul</span>(<span class="pl-s1">embedding_after_edit_normalized</span>, <span class="pl-s1">w3</span>.<span class="pl-v">T</span>), <span class="pl-s1">w2</span>.<span class="pl-v">T</span>)
<span class="pl-s1">output_after_feedforward</span>.<span class="pl-s1">shape</span></pre><div class="zeroclipboard-container">
    
  </div></div>
<div class="snippet-clipboard-content notranslate position-relative overflow-auto"><pre class="notranslate"><code>torch.Size([17, 4096])
</code></pre><div class="zeroclipboard-container">
    
  </div></div>
<div class="markdown-heading" dir="auto"><h1 tabindex="-1" class="heading-element" dir="auto" _msttexthash="133293355" _msthash="297">我们终于为第一层之后的每个 TOKEN 有了新的编辑嵌入</h1><a id="user-content-we-finally-have-new-edited-embeddings-for-each-token-after-the-first-layer" class="anchor" aria-label="永久链接：我们终于为第一层之后的每个 TOKEN 提供了新的编辑嵌入" href="#we-finally-have-new-edited-embeddings-for-each-token-after-the-first-layer" _mstaria-label="3009188" _msthash="298"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></div>
<p dir="auto" _msttexthash="2388730331" _msthash="299">在我们完成之前，只剩下 31 层了（一个 for 循环），<br _istranslated="1">你可以想象这个编辑后的嵌入包含第一层<br _istranslated="1">上提出的所有查询的信息，现在每一层都会对提出的问题进行越来越复杂的查询编码，直到我们有一个知道我们需要的下一个标记的所有信息的嵌入。</p>
<div class="highlight highlight-source-python notranslate position-relative overflow-auto" dir="auto"><pre><span class="pl-s1">layer_0_embedding</span> <span class="pl-c1">=</span> <span class="pl-s1">embedding_after_edit</span><span class="pl-c1">+</span><span class="pl-s1">output_after_feedforward</span>
<span class="pl-s1">layer_0_embedding</span>.<span class="pl-s1">shape</span></pre><div class="zeroclipboard-container">
    
  </div></div>
<div class="snippet-clipboard-content notranslate position-relative overflow-auto"><pre class="notranslate"><code>torch.Size([17, 4096])
</code></pre><div class="zeroclipboard-container">
     
  </div></div>
<div class="markdown-heading" dir="auto"><h1 tabindex="-1" class="heading-element" dir="auto" _msttexthash="42159130" _msthash="300">上帝，一切都同时发生</h1><a id="user-content-god-everything-all-at-once" class="anchor" aria-label="永久链接： 天哪，一切都同时发生" href="#god-everything-all-at-once" _mstaria-label="1024543" _msthash="301"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></div>
<div dir="auto">
    <a target="_blank" rel="noopener noreferrer" href="https://github.com/naklecha/llama3-from-scratch/blob/main/images/god.png"><img src="https://github.com/naklecha/llama3-from-scratch/raw/main/images/god.png" width="600px" style="max-width: 100%;"></a>
</div><font _mstmutation="1" _msttexthash="227482983" _msthash="302">是的，就是这个。我们之前所做的一切，一次性完成，用于每一层。</font><br>
<div class="markdown-heading" dir="auto"><h1 tabindex="-1" class="heading-element" dir="auto" _msttexthash="21952697" _msthash="303">祝您阅读愉快 :)</h1><a id="user-content-have-fun-reading-" class="anchor" aria-label="永久链接：祝您阅读愉快 :)" href="#have-fun-reading-" _mstaria-label="638560" _msthash="304"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></div>
<div class="highlight highlight-source-python notranslate position-relative overflow-auto" dir="auto"><pre><span class="pl-s1">final_embedding</span> <span class="pl-c1">=</span> <span class="pl-s1">token_embeddings_unnormalized</span>
<span class="pl-k">for</span> <span class="pl-s1">layer</span> <span class="pl-c1">in</span> <span class="pl-en">range</span>(<span class="pl-s1">n_layers</span>):
    <span class="pl-s1">qkv_attention_store</span> <span class="pl-c1">=</span> []
    <span class="pl-s1">layer_embedding_norm</span> <span class="pl-c1">=</span> <span class="pl-en">rms_norm</span>(<span class="pl-s1">final_embedding</span>, <span class="pl-s1">model</span>[<span class="pl-s">f"layers.<span class="pl-s1"><span class="pl-kos">{</span><span class="pl-s1">layer</span><span class="pl-kos">}</span></span>.attention_norm.weight"</span>])
    <span class="pl-s1">q_layer</span> <span class="pl-c1">=</span> <span class="pl-s1">model</span>[<span class="pl-s">f"layers.<span class="pl-s1"><span class="pl-kos">{</span><span class="pl-s1">layer</span><span class="pl-kos">}</span></span>.attention.wq.weight"</span>]
    <span class="pl-s1">q_layer</span> <span class="pl-c1">=</span> <span class="pl-s1">q_layer</span>.<span class="pl-en">view</span>(<span class="pl-s1">n_heads</span>, <span class="pl-s1">q_layer</span>.<span class="pl-s1">shape</span>[<span class="pl-c1">0</span>] <span class="pl-c1">//</span> <span class="pl-s1">n_heads</span>, <span class="pl-s1">dim</span>)
    <span class="pl-s1">k_layer</span> <span class="pl-c1">=</span> <span class="pl-s1">model</span>[<span class="pl-s">f"layers.<span class="pl-s1"><span class="pl-kos">{</span><span class="pl-s1">layer</span><span class="pl-kos">}</span></span>.attention.wk.weight"</span>]
    <span class="pl-s1">k_layer</span> <span class="pl-c1">=</span> <span class="pl-s1">k_layer</span>.<span class="pl-en">view</span>(<span class="pl-s1">n_kv_heads</span>, <span class="pl-s1">k_layer</span>.<span class="pl-s1">shape</span>[<span class="pl-c1">0</span>] <span class="pl-c1">//</span> <span class="pl-s1">n_kv_heads</span>, <span class="pl-s1">dim</span>)
    <span class="pl-s1">v_layer</span> <span class="pl-c1">=</span> <span class="pl-s1">model</span>[<span class="pl-s">f"layers.<span class="pl-s1"><span class="pl-kos">{</span><span class="pl-s1">layer</span><span class="pl-kos">}</span></span>.attention.wv.weight"</span>]
    <span class="pl-s1">v_layer</span> <span class="pl-c1">=</span> <span class="pl-s1">v_layer</span>.<span class="pl-en">view</span>(<span class="pl-s1">n_kv_heads</span>, <span class="pl-s1">v_layer</span>.<span class="pl-s1">shape</span>[<span class="pl-c1">0</span>] <span class="pl-c1">//</span> <span class="pl-s1">n_kv_heads</span>, <span class="pl-s1">dim</span>)
    <span class="pl-s1">w_layer</span> <span class="pl-c1">=</span> <span class="pl-s1">model</span>[<span class="pl-s">f"layers.<span class="pl-s1"><span class="pl-kos">{</span><span class="pl-s1">layer</span><span class="pl-kos">}</span></span>.attention.wo.weight"</span>]
    <span class="pl-k">for</span> <span class="pl-s1">head</span> <span class="pl-c1">in</span> <span class="pl-en">range</span>(<span class="pl-s1">n_heads</span>):
        <span class="pl-s1">q_layer_head</span> <span class="pl-c1">=</span> <span class="pl-s1">q_layer</span>[<span class="pl-s1">head</span>]
        <span class="pl-s1">k_layer_head</span> <span class="pl-c1">=</span> <span class="pl-s1">k_layer</span>[<span class="pl-s1">head</span><span class="pl-c1">//</span><span class="pl-c1">4</span>]
        <span class="pl-s1">v_layer_head</span> <span class="pl-c1">=</span> <span class="pl-s1">v_layer</span>[<span class="pl-s1">head</span><span class="pl-c1">//</span><span class="pl-c1">4</span>]
        <span class="pl-s1">q_per_token</span> <span class="pl-c1">=</span> <span class="pl-s1">torch</span>.<span class="pl-en">matmul</span>(<span class="pl-s1">layer_embedding_norm</span>, <span class="pl-s1">q_layer_head</span>.<span class="pl-v">T</span>)
        <span class="pl-s1">k_per_token</span> <span class="pl-c1">=</span> <span class="pl-s1">torch</span>.<span class="pl-en">matmul</span>(<span class="pl-s1">layer_embedding_norm</span>, <span class="pl-s1">k_layer_head</span>.<span class="pl-v">T</span>)
        <span class="pl-s1">v_per_token</span> <span class="pl-c1">=</span> <span class="pl-s1">torch</span>.<span class="pl-en">matmul</span>(<span class="pl-s1">layer_embedding_norm</span>, <span class="pl-s1">v_layer_head</span>.<span class="pl-v">T</span>)
        <span class="pl-s1">q_per_token_split_into_pairs</span> <span class="pl-c1">=</span> <span class="pl-s1">q_per_token</span>.<span class="pl-en">float</span>().<span class="pl-en">view</span>(<span class="pl-s1">q_per_token</span>.<span class="pl-s1">shape</span>[<span class="pl-c1">0</span>], <span class="pl-c1">-</span><span class="pl-c1">1</span>, <span class="pl-c1">2</span>)
        <span class="pl-s1">q_per_token_as_complex_numbers</span> <span class="pl-c1">=</span> <span class="pl-s1">torch</span>.<span class="pl-en">view_as_complex</span>(<span class="pl-s1">q_per_token_split_into_pairs</span>)
        <span class="pl-s1">q_per_token_split_into_pairs_rotated</span> <span class="pl-c1">=</span> <span class="pl-s1">torch</span>.<span class="pl-en">view_as_real</span>(<span class="pl-s1">q_per_token_as_complex_numbers</span> <span class="pl-c1">*</span> <span class="pl-s1">freqs_cis</span>)
        <span class="pl-s1">q_per_token_rotated</span> <span class="pl-c1">=</span> <span class="pl-s1">q_per_token_split_into_pairs_rotated</span>.<span class="pl-en">view</span>(<span class="pl-s1">q_per_token</span>.<span class="pl-s1">shape</span>)
        <span class="pl-s1">k_per_token_split_into_pairs</span> <span class="pl-c1">=</span> <span class="pl-s1">k_per_token</span>.<span class="pl-en">float</span>().<span class="pl-en">view</span>(<span class="pl-s1">k_per_token</span>.<span class="pl-s1">shape</span>[<span class="pl-c1">0</span>], <span class="pl-c1">-</span><span class="pl-c1">1</span>, <span class="pl-c1">2</span>)
        <span class="pl-s1">k_per_token_as_complex_numbers</span> <span class="pl-c1">=</span> <span class="pl-s1">torch</span>.<span class="pl-en">view_as_complex</span>(<span class="pl-s1">k_per_token_split_into_pairs</span>)
        <span class="pl-s1">k_per_token_split_into_pairs_rotated</span> <span class="pl-c1">=</span> <span class="pl-s1">torch</span>.<span class="pl-en">view_as_real</span>(<span class="pl-s1">k_per_token_as_complex_numbers</span> <span class="pl-c1">*</span> <span class="pl-s1">freqs_cis</span>)
        <span class="pl-s1">k_per_token_rotated</span> <span class="pl-c1">=</span> <span class="pl-s1">k_per_token_split_into_pairs_rotated</span>.<span class="pl-en">view</span>(<span class="pl-s1">k_per_token</span>.<span class="pl-s1">shape</span>)
        <span class="pl-s1">qk_per_token</span> <span class="pl-c1">=</span> <span class="pl-s1">torch</span>.<span class="pl-en">matmul</span>(<span class="pl-s1">q_per_token_rotated</span>, <span class="pl-s1">k_per_token_rotated</span>.<span class="pl-v">T</span>)<span class="pl-c1">/</span>(<span class="pl-c1">128</span>)<span class="pl-c1">**</span><span class="pl-c1">0.5</span>
        <span class="pl-s1">mask</span> <span class="pl-c1">=</span> <span class="pl-s1">torch</span>.<span class="pl-en">full</span>((<span class="pl-en">len</span>(<span class="pl-s1">token_embeddings_unnormalized</span>), <span class="pl-en">len</span>(<span class="pl-s1">token_embeddings_unnormalized</span>)), <span class="pl-en">float</span>(<span class="pl-s">"-inf"</span>))
        <span class="pl-s1">mask</span> <span class="pl-c1">=</span> <span class="pl-s1">torch</span>.<span class="pl-en">triu</span>(<span class="pl-s1">mask</span>, <span class="pl-s1">diagonal</span><span class="pl-c1">=</span><span class="pl-c1">1</span>)
        <span class="pl-s1">qk_per_token_after_masking</span> <span class="pl-c1">=</span> <span class="pl-s1">qk_per_token</span> <span class="pl-c1">+</span> <span class="pl-s1">mask</span>
        <span class="pl-s1">qk_per_token_after_masking_after_softmax</span> <span class="pl-c1">=</span> <span class="pl-s1">torch</span>.<span class="pl-s1">nn</span>.<span class="pl-s1">functional</span>.<span class="pl-en">softmax</span>(<span class="pl-s1">qk_per_token_after_masking</span>, <span class="pl-s1">dim</span><span class="pl-c1">=</span><span class="pl-c1">1</span>).<span class="pl-en">to</span>(<span class="pl-s1">torch</span>.<span class="pl-s1">bfloat16</span>)
        <span class="pl-s1">qkv_attention</span> <span class="pl-c1">=</span> <span class="pl-s1">torch</span>.<span class="pl-en">matmul</span>(<span class="pl-s1">qk_per_token_after_masking_after_softmax</span>, <span class="pl-s1">v_per_token</span>)
        <span class="pl-s1">qkv_attention_store</span>.<span class="pl-en">append</span>(<span class="pl-s1">qkv_attention</span>)

    <span class="pl-s1">stacked_qkv_attention</span> <span class="pl-c1">=</span> <span class="pl-s1">torch</span>.<span class="pl-en">cat</span>(<span class="pl-s1">qkv_attention_store</span>, <span class="pl-s1">dim</span><span class="pl-c1">=</span><span class="pl-c1">-</span><span class="pl-c1">1</span>)
    <span class="pl-s1">w_layer</span> <span class="pl-c1">=</span> <span class="pl-s1">model</span>[<span class="pl-s">f"layers.<span class="pl-s1"><span class="pl-kos">{</span><span class="pl-s1">layer</span><span class="pl-kos">}</span></span>.attention.wo.weight"</span>]
    <span class="pl-s1">embedding_delta</span> <span class="pl-c1">=</span> <span class="pl-s1">torch</span>.<span class="pl-en">matmul</span>(<span class="pl-s1">stacked_qkv_attention</span>, <span class="pl-s1">w_layer</span>.<span class="pl-v">T</span>)
    <span class="pl-s1">embedding_after_edit</span> <span class="pl-c1">=</span> <span class="pl-s1">final_embedding</span> <span class="pl-c1">+</span> <span class="pl-s1">embedding_delta</span>
    <span class="pl-s1">embedding_after_edit_normalized</span> <span class="pl-c1">=</span> <span class="pl-en">rms_norm</span>(<span class="pl-s1">embedding_after_edit</span>, <span class="pl-s1">model</span>[<span class="pl-s">f"layers.<span class="pl-s1"><span class="pl-kos">{</span><span class="pl-s1">layer</span><span class="pl-kos">}</span></span>.ffn_norm.weight"</span>])
    <span class="pl-s1">w1</span> <span class="pl-c1">=</span> <span class="pl-s1">model</span>[<span class="pl-s">f"layers.<span class="pl-s1"><span class="pl-kos">{</span><span class="pl-s1">layer</span><span class="pl-kos">}</span></span>.feed_forward.w1.weight"</span>]
    <span class="pl-s1">w2</span> <span class="pl-c1">=</span> <span class="pl-s1">model</span>[<span class="pl-s">f"layers.<span class="pl-s1"><span class="pl-kos">{</span><span class="pl-s1">layer</span><span class="pl-kos">}</span></span>.feed_forward.w2.weight"</span>]
    <span class="pl-s1">w3</span> <span class="pl-c1">=</span> <span class="pl-s1">model</span>[<span class="pl-s">f"layers.<span class="pl-s1"><span class="pl-kos">{</span><span class="pl-s1">layer</span><span class="pl-kos">}</span></span>.feed_forward.w3.weight"</span>]
    <span class="pl-s1">output_after_feedforward</span> <span class="pl-c1">=</span> <span class="pl-s1">torch</span>.<span class="pl-en">matmul</span>(<span class="pl-s1">torch</span>.<span class="pl-s1">functional</span>.<span class="pl-v">F</span>.<span class="pl-en">silu</span>(<span class="pl-s1">torch</span>.<span class="pl-en">matmul</span>(<span class="pl-s1">embedding_after_edit_normalized</span>, <span class="pl-s1">w1</span>.<span class="pl-v">T</span>)) <span class="pl-c1">*</span> <span class="pl-s1">torch</span>.<span class="pl-en">matmul</span>(<span class="pl-s1">embedding_after_edit_normalized</span>, <span class="pl-s1">w3</span>.<span class="pl-v">T</span>), <span class="pl-s1">w2</span>.<span class="pl-v">T</span>)
    <span class="pl-s1">final_embedding</span> <span class="pl-c1">=</span> <span class="pl-s1">embedding_after_edit</span><span class="pl-c1">+</span><span class="pl-s1">output_after_feedforward</span></pre><div class="zeroclipboard-container">
    <clipboard-copy aria-label="Copy" class="ClipboardButton btn btn-invisible js-clipboard-copy m-2 p-0 d-flex flex-justify-center flex-items-center" data-copy-feedback="Copied!" data-tooltip-direction="w" value="final_embedding = token_embeddings_unnormalized
for layer in range(n_layers):
    qkv_attention_store = []
    layer_embedding_norm = rms_norm(final_embedding, model[f&quot;layers.{layer}.attention_norm.weight&quot;])
    q_layer = model[f&quot;layers.{layer}.attention.wq.weight&quot;]
    q_layer = q_layer.view(n_heads, q_layer.shape[0] // n_heads, dim)
    k_layer = model[f&quot;layers.{layer}.attention.wk.weight&quot;]
    k_layer = k_layer.view(n_kv_heads, k_layer.shape[0] // n_kv_heads, dim)
    v_layer = model[f&quot;layers.{layer}.attention.wv.weight&quot;]
    v_layer = v_layer.view(n_kv_heads, v_layer.shape[0] // n_kv_heads, dim)
    w_layer = model[f&quot;layers.{layer}.attention.wo.weight&quot;]
    for head in range(n_heads):
        q_layer_head = q_layer[head]
        k_layer_head = k_layer[head//4]
        v_layer_head = v_layer[head//4]
        q_per_token = torch.matmul(layer_embedding_norm, q_layer_head.T)
        k_per_token = torch.matmul(layer_embedding_norm, k_layer_head.T)
        v_per_token = torch.matmul(layer_embedding_norm, v_layer_head.T)
        q_per_token_split_into_pairs = q_per_token.float().view(q_per_token.shape[0], -1, 2)
        q_per_token_as_complex_numbers = torch.view_as_complex(q_per_token_split_into_pairs)
        q_per_token_split_into_pairs_rotated = torch.view_as_real(q_per_token_as_complex_numbers * freqs_cis)
        q_per_token_rotated = q_per_token_split_into_pairs_rotated.view(q_per_token.shape)
        k_per_token_split_into_pairs = k_per_token.float().view(k_per_token.shape[0], -1, 2)
        k_per_token_as_complex_numbers = torch.view_as_complex(k_per_token_split_into_pairs)
        k_per_token_split_into_pairs_rotated = torch.view_as_real(k_per_token_as_complex_numbers * freqs_cis)
        k_per_token_rotated = k_per_token_split_into_pairs_rotated.view(k_per_token.shape)
        qk_per_token = torch.matmul(q_per_token_rotated, k_per_token_rotated.T)/(128)**0.5
        mask = torch.full((len(token_embeddings_unnormalized), len(token_embeddings_unnormalized)), float(&quot;-inf&quot;))
        mask = torch.triu(mask, diagonal=1)
        qk_per_token_after_masking = qk_per_token + mask
        qk_per_token_after_masking_after_softmax = torch.nn.functional.softmax(qk_per_token_after_masking, dim=1).to(torch.bfloat16)
        qkv_attention = torch.matmul(qk_per_token_after_masking_after_softmax, v_per_token)
        qkv_attention_store.append(qkv_attention)

    stacked_qkv_attention = torch.cat(qkv_attention_store, dim=-1)
    w_layer = model[f&quot;layers.{layer}.attention.wo.weight&quot;]
    embedding_delta = torch.matmul(stacked_qkv_attention, w_layer.T)
    embedding_after_edit = final_embedding + embedding_delta
    embedding_after_edit_normalized = rms_norm(embedding_after_edit, model[f&quot;layers.{layer}.ffn_norm.weight&quot;])
    w1 = model[f&quot;layers.{layer}.feed_forward.w1.weight&quot;]
    w2 = model[f&quot;layers.{layer}.feed_forward.w2.weight&quot;]
    w3 = model[f&quot;layers.{layer}.feed_forward.w3.weight&quot;]
    output_after_feedforward = torch.matmul(torch.functional.F.silu(torch.matmul(embedding_after_edit_normalized, w1.T)) * torch.matmul(embedding_after_edit_normalized, w3.T), w2.T)
    final_embedding = embedding_after_edit+output_after_feedforward" tabindex="0" role="button">
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-copy js-clipboard-copy-icon">
    <path d="M0 6.75C0 5.784.784 5 1.75 5h1.5a.75.75 0 0 1 0 1.5h-1.5a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-1.5a.75.75 0 0 1 1.5 0v1.5A1.75 1.75 0 0 1 9.25 16h-7.5A1.75 1.75 0 0 1 0 14.25Z"></path><path d="M5 1.75C5 .784 5.784 0 6.75 0h7.5C15.216 0 16 .784 16 1.75v7.5A1.75 1.75 0 0 1 14.25 11h-7.5A1.75 1.75 0 0 1 5 9.25Zm1.75-.25a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-7.5a.25.25 0 0 0-.25-.25Z"></path>
</svg>
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-check js-clipboard-check-icon color-fg-success d-none">
    <path d="M13.78 4.22a.75.75 0 0 1 0 1.06l-7.25 7.25a.75.75 0 0 1-1.06 0L2.22 9.28a.751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018L6 10.94l6.72-6.72a.75.75 0 0 1 1.06 0Z"></path>
</svg>
    </clipboard-copy>
  </div></div>
<div class="markdown-heading" dir="auto"><h1 tabindex="-1" class="heading-element" dir="auto" _msttexthash="174659277" _msthash="305">我们现在有了最终的嵌入，这是模型对下一个 Token 的最佳猜测</h1><a id="user-content-we-now-have-the-final-embedding-the-best-guess-the-model-could-make-about-the-next-token" class="anchor" aria-label="永久链接：我们现在有了最终的嵌入，这是模型对下一个 token 的最佳猜测" href="#we-now-have-the-final-embedding-the-best-guess-the-model-could-make-about-the-next-token" _mstaria-label="5505214" _msthash="306"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></div>
<p dir="auto" _msttexthash="294020324" _msthash="307">嵌入的形状与常规标记嵌入相同 [17x4096]，其中 17 是标记的数量，4096 是嵌入的 Dim</p>
<div dir="auto">
    <a target="_blank" rel="noopener noreferrer" href="/naklecha/llama3-from-scratch/blob/main/images/last_norm.png"><img src="/naklecha/llama3-from-scratch/raw/main/images/last_norm.png" width="600px" style="max-width: 100%;"></a>
</div>
<div class="highlight highlight-source-python notranslate position-relative overflow-auto" dir="auto"><pre><span class="pl-s1">final_embedding</span> <span class="pl-c1">=</span> <span class="pl-en">rms_norm</span>(<span class="pl-s1">final_embedding</span>, <span class="pl-s1">model</span>[<span class="pl-s">"norm.weight"</span>])
<span class="pl-s1">final_embedding</span>.<span class="pl-s1">shape</span></pre><div class="zeroclipboard-container">
    <clipboard-copy aria-label="Copy" class="ClipboardButton btn btn-invisible js-clipboard-copy m-2 p-0 d-flex flex-justify-center flex-items-center" data-copy-feedback="Copied!" data-tooltip-direction="w" value="final_embedding = rms_norm(final_embedding, model[&quot;norm.weight&quot;])
final_embedding.shape" tabindex="0" role="button">
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-copy js-clipboard-copy-icon">
    <path d="M0 6.75C0 5.784.784 5 1.75 5h1.5a.75.75 0 0 1 0 1.5h-1.5a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-1.5a.75.75 0 0 1 1.5 0v1.5A1.75 1.75 0 0 1 9.25 16h-7.5A1.75 1.75 0 0 1 0 14.25Z"></path><path d="M5 1.75C5 .784 5.784 0 6.75 0h7.5C15.216 0 16 .784 16 1.75v7.5A1.75 1.75 0 0 1 14.25 11h-7.5A1.75 1.75 0 0 1 5 9.25Zm1.75-.25a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-7.5a.25.25 0 0 0-.25-.25Z"></path>
</svg>
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-check js-clipboard-check-icon color-fg-success d-none">
    <path d="M13.78 4.22a.75.75 0 0 1 0 1.06l-7.25 7.25a.75.75 0 0 1-1.06 0L2.22 9.28a.751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018L6 10.94l6.72-6.72a.75.75 0 0 1 1.06 0Z"></path>
</svg>
    </clipboard-copy>
  </div></div>
<div class="snippet-clipboard-content notranslate position-relative overflow-auto"><pre class="notranslate"><code>torch.Size([17, 4096])
</code></pre><div class="zeroclipboard-container">
    <clipboard-copy aria-label="Copy" class="ClipboardButton btn btn-invisible js-clipboard-copy m-2 p-0 d-flex flex-justify-center flex-items-center" data-copy-feedback="Copied!" data-tooltip-direction="w" value="torch.Size([17, 4096])" tabindex="0" role="button">
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-copy js-clipboard-copy-icon">
    <path d="M0 6.75C0 5.784.784 5 1.75 5h1.5a.75.75 0 0 1 0 1.5h-1.5a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-1.5a.75.75 0 0 1 1.5 0v1.5A1.75 1.75 0 0 1 9.25 16h-7.5A1.75 1.75 0 0 1 0 14.25Z"></path><path d="M5 1.75C5 .784 5.784 0 6.75 0h7.5C15.216 0 16 .784 16 1.75v7.5A1.75 1.75 0 0 1 14.25 11h-7.5A1.75 1.75 0 0 1 5 9.25Zm1.75-.25a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-7.5a.25.25 0 0 0-.25-.25Z"></path>
</svg>
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-check js-clipboard-check-icon color-fg-success d-none">
    <path d="M13.78 4.22a.75.75 0 0 1 0 1.06l-7.25 7.25a.75.75 0 0 1-1.06 0L2.22 9.28a.751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018L6 10.94l6.72-6.72a.75.75 0 0 1 1.06 0Z"></path>
</svg>
    </clipboard-copy>
  </div></div>
<div class="markdown-heading" dir="auto"><h1 tabindex="-1" class="heading-element" dir="auto" _msttexthash="61424350" _msthash="308">最后，让我们将嵌入解码为 Token 值</h1><a id="user-content-finally-lets-decode-the-embedding-into-the-token-value" class="anchor" aria-label="永久链接：最后，让我们将 embedding 解码为 token 值" href="#finally-lets-decode-the-embedding-into-the-token-value" _mstaria-label="2736435" _msthash="309"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></div>
<div dir="auto">
    <a target="_blank" rel="noopener noreferrer" href="/naklecha/llama3-from-scratch/blob/main/images/finallayer.png"><img src="/naklecha/llama3-from-scratch/raw/main/images/finallayer.png" width="600px" style="max-width: 100%;"></a>
</div><font _mstmutation="1" _msttexthash="106751385" _msthash="310">我们将使用 output 解码器将最终的 embedding 转换为 token</font><div class="highlight highlight-source-python notranslate position-relative overflow-auto" dir="auto"><pre><span class="pl-s1">model</span>[<span class="pl-s">"output.weight"</span>].<span class="pl-s1">shape</span></pre><div class="zeroclipboard-container">
    <clipboard-copy aria-label="Copy" class="ClipboardButton btn btn-invisible js-clipboard-copy m-2 p-0 d-flex flex-justify-center flex-items-center" data-copy-feedback="Copied!" data-tooltip-direction="w" value="model[&quot;output.weight&quot;].shape" tabindex="0" role="button">
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-copy js-clipboard-copy-icon">
    <path d="M0 6.75C0 5.784.784 5 1.75 5h1.5a.75.75 0 0 1 0 1.5h-1.5a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-1.5a.75.75 0 0 1 1.5 0v1.5A1.75 1.75 0 0 1 9.25 16h-7.5A1.75 1.75 0 0 1 0 14.25Z"></path><path d="M5 1.75C5 .784 5.784 0 6.75 0h7.5C15.216 0 16 .784 16 1.75v7.5A1.75 1.75 0 0 1 14.25 11h-7.5A1.75 1.75 0 0 1 5 9.25Zm1.75-.25a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-7.5a.25.25 0 0 0-.25-.25Z"></path>
</svg>
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-check js-clipboard-check-icon color-fg-success d-none">
    <path d="M13.78 4.22a.75.75 0 0 1 0 1.06l-7.25 7.25a.75.75 0 0 1-1.06 0L2.22 9.28a.751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018L6 10.94l6.72-6.72a.75.75 0 0 1 1.06 0Z"></path>
</svg>
    </clipboard-copy>
  </div></div>
<div class="snippet-clipboard-content notranslate position-relative overflow-auto"><pre class="notranslate"><code>torch.Size([128256, 4096])
</code></pre><div class="zeroclipboard-container">
    <clipboard-copy aria-label="Copy" class="ClipboardButton btn btn-invisible js-clipboard-copy m-2 p-0 d-flex flex-justify-center flex-items-center" data-copy-feedback="Copied!" data-tooltip-direction="w" value="torch.Size([128256, 4096])" tabindex="0" role="button">
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-copy js-clipboard-copy-icon">
    <path d="M0 6.75C0 5.784.784 5 1.75 5h1.5a.75.75 0 0 1 0 1.5h-1.5a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-1.5a.75.75 0 0 1 1.5 0v1.5A1.75 1.75 0 0 1 9.25 16h-7.5A1.75 1.75 0 0 1 0 14.25Z"></path><path d="M5 1.75C5 .784 5.784 0 6.75 0h7.5C15.216 0 16 .784 16 1.75v7.5A1.75 1.75 0 0 1 14.25 11h-7.5A1.75 1.75 0 0 1 5 9.25Zm1.75-.25a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-7.5a.25.25 0 0 0-.25-.25Z"></path>
</svg>
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-check js-clipboard-check-icon color-fg-success d-none">
    <path d="M13.78 4.22a.75.75 0 0 1 0 1.06l-7.25 7.25a.75.75 0 0 1-1.06 0L2.22 9.28a.751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018L6 10.94l6.72-6.72a.75.75 0 0 1 1.06 0Z"></path>
</svg>
    </clipboard-copy>
  </div></div>
<div class="markdown-heading" dir="auto"><h1 tabindex="-1" class="heading-element" dir="auto" _msttexthash="106073955" _msthash="311">我们使用最后一个标记的嵌入来预测下一个值</h1><a id="user-content-we-use-the-embedding-of-the-last-token-to-predict-the-next-value" class="anchor" aria-label="永久链接：我们使用最后一个 token 的 embedding 来预测下一个值" href="#we-use-the-embedding-of-the-last-token-to-predict-the-next-value" _mstaria-label="3247673" _msthash="312"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></div>
<p dir="auto" _msttexthash="1788901062" _msthash="313">希望在我们的例子中，42 :)
注意：42 是“生命、宇宙和万物的终极问题的答案是”的答案，根据《银河系漫游指南》一书，大多数现代 LLM 都会在这里用 42 来回答，这应该验证了我们的整个代码！祝我好运:)</p>
<div class="highlight highlight-source-python notranslate position-relative overflow-auto" dir="auto"><pre><span class="pl-s1">logits</span> <span class="pl-c1">=</span> <span class="pl-s1">torch</span>.<span class="pl-en">matmul</span>(<span class="pl-s1">final_embedding</span>[<span class="pl-c1">-</span><span class="pl-c1">1</span>], <span class="pl-s1">model</span>[<span class="pl-s">"output.weight"</span>].<span class="pl-v">T</span>)
<span class="pl-s1">logits</span>.<span class="pl-s1">shape</span></pre><div class="zeroclipboard-container">
    <clipboard-copy aria-label="Copy" class="ClipboardButton btn btn-invisible js-clipboard-copy m-2 p-0 d-flex flex-justify-center flex-items-center" data-copy-feedback="Copied!" data-tooltip-direction="w" value="logits = torch.matmul(final_embedding[-1], model[&quot;output.weight&quot;].T)
logits.shape" tabindex="0" role="button">
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-copy js-clipboard-copy-icon">
    <path d="M0 6.75C0 5.784.784 5 1.75 5h1.5a.75.75 0 0 1 0 1.5h-1.5a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-1.5a.75.75 0 0 1 1.5 0v1.5A1.75 1.75 0 0 1 9.25 16h-7.5A1.75 1.75 0 0 1 0 14.25Z"></path><path d="M5 1.75C5 .784 5.784 0 6.75 0h7.5C15.216 0 16 .784 16 1.75v7.5A1.75 1.75 0 0 1 14.25 11h-7.5A1.75 1.75 0 0 1 5 9.25Zm1.75-.25a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-7.5a.25.25 0 0 0-.25-.25Z"></path>
</svg>
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-check js-clipboard-check-icon color-fg-success d-none">
    <path d="M13.78 4.22a.75.75 0 0 1 0 1.06l-7.25 7.25a.75.75 0 0 1-1.06 0L2.22 9.28a.751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018L6 10.94l6.72-6.72a.75.75 0 0 1 1.06 0Z"></path>
</svg>
    </clipboard-copy>
  </div></div>
<div class="snippet-clipboard-content notranslate position-relative overflow-auto"><pre class="notranslate"><code>torch.Size([128256])
</code></pre><div class="zeroclipboard-container">
    <clipboard-copy aria-label="Copy" class="ClipboardButton btn btn-invisible js-clipboard-copy m-2 p-0 d-flex flex-justify-center flex-items-center" data-copy-feedback="Copied!" data-tooltip-direction="w" value="torch.Size([128256])" tabindex="0" role="button">
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-copy js-clipboard-copy-icon">
    <path d="M0 6.75C0 5.784.784 5 1.75 5h1.5a.75.75 0 0 1 0 1.5h-1.5a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-1.5a.75.75 0 0 1 1.5 0v1.5A1.75 1.75 0 0 1 9.25 16h-7.5A1.75 1.75 0 0 1 0 14.25Z"></path><path d="M5 1.75C5 .784 5.784 0 6.75 0h7.5C15.216 0 16 .784 16 1.75v7.5A1.75 1.75 0 0 1 14.25 11h-7.5A1.75 1.75 0 0 1 5 9.25Zm1.75-.25a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-7.5a.25.25 0 0 0-.25-.25Z"></path>
</svg>
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-check js-clipboard-check-icon color-fg-success d-none">
    <path d="M13.78 4.22a.75.75 0 0 1 0 1.06l-7.25 7.25a.75.75 0 0 1-1.06 0L2.22 9.28a.751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018L6 10.94l6.72-6.72a.75.75 0 0 1 1.06 0Z"></path>
</svg>
    </clipboard-copy>
  </div></div>
<div class="markdown-heading" dir="auto"><h3 tabindex="-1" class="heading-element" dir="auto" _msttexthash="211758456" _msthash="314">模型预测代币编号 2983 作为下一个代币，这是 42 的代币编号吗？</h3><a id="user-content-the-model-predicted-token-number-2983-as-the-next-token-is-this-the-token-number-for-42" class="anchor" aria-label="永久链接：模型预测 Token 编号 2983 作为下一个 Token，这是 42 的 Token 编号吗？" href="#the-model-predicted-token-number-2983-as-the-next-token-is-this-the-token-number-for-42" _mstaria-label="5227495" _msthash="315"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></div>
<p dir="auto" _msttexthash="147686513" _msthash="316">我提醒你，这是最后一部分代码，希望你:)玩得开心</p>
<div class="highlight highlight-source-python notranslate position-relative overflow-auto" dir="auto"><pre><span class="pl-s1">next_token</span> <span class="pl-c1">=</span> <span class="pl-s1">torch</span>.<span class="pl-en">argmax</span>(<span class="pl-s1">logits</span>, <span class="pl-s1">dim</span><span class="pl-c1">=</span><span class="pl-c1">-</span><span class="pl-c1">1</span>)
<span class="pl-s1">next_token</span></pre><div class="zeroclipboard-container">
    <clipboard-copy aria-label="Copy" class="ClipboardButton btn btn-invisible js-clipboard-copy m-2 p-0 d-flex flex-justify-center flex-items-center" data-copy-feedback="Copied!" data-tooltip-direction="w" value="next_token = torch.argmax(logits, dim=-1)
next_token" tabindex="0" role="button">
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-copy js-clipboard-copy-icon">
    <path d="M0 6.75C0 5.784.784 5 1.75 5h1.5a.75.75 0 0 1 0 1.5h-1.5a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-1.5a.75.75 0 0 1 1.5 0v1.5A1.75 1.75 0 0 1 9.25 16h-7.5A1.75 1.75 0 0 1 0 14.25Z"></path><path d="M5 1.75C5 .784 5.784 0 6.75 0h7.5C15.216 0 16 .784 16 1.75v7.5A1.75 1.75 0 0 1 14.25 11h-7.5A1.75 1.75 0 0 1 5 9.25Zm1.75-.25a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-7.5a.25.25 0 0 0-.25-.25Z"></path>
</svg>
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-check js-clipboard-check-icon color-fg-success d-none">
    <path d="M13.78 4.22a.75.75 0 0 1 0 1.06l-7.25 7.25a.75.75 0 0 1-1.06 0L2.22 9.28a.751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018L6 10.94l6.72-6.72a.75.75 0 0 1 1.06 0Z"></path>
</svg>
    </clipboard-copy>
  </div></div>
<div class="snippet-clipboard-content notranslate position-relative overflow-auto"><pre class="notranslate"><code>tensor(2983)
</code></pre><div class="zeroclipboard-container">
    <clipboard-copy aria-label="Copy" class="ClipboardButton btn btn-invisible js-clipboard-copy m-2 p-0 d-flex flex-justify-center flex-items-center" data-copy-feedback="Copied!" data-tooltip-direction="w" value="tensor(2983)" tabindex="0" role="button">
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-copy js-clipboard-copy-icon">
    <path d="M0 6.75C0 5.784.784 5 1.75 5h1.5a.75.75 0 0 1 0 1.5h-1.5a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-1.5a.75.75 0 0 1 1.5 0v1.5A1.75 1.75 0 0 1 9.25 16h-7.5A1.75 1.75 0 0 1 0 14.25Z"></path><path d="M5 1.75C5 .784 5.784 0 6.75 0h7.5C15.216 0 16 .784 16 1.75v7.5A1.75 1.75 0 0 1 14.25 11h-7.5A1.75 1.75 0 0 1 5 9.25Zm1.75-.25a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-7.5a.25.25 0 0 0-.25-.25Z"></path>
</svg>
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-check js-clipboard-check-icon color-fg-success d-none">
    <path d="M13.78 4.22a.75.75 0 0 1 0 1.06l-7.25 7.25a.75.75 0 0 1-1.06 0L2.22 9.28a.751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018L6 10.94l6.72-6.72a.75.75 0 0 1 1.06 0Z"></path>
</svg>
    </clipboard-copy>
  </div></div>
<div class="markdown-heading" dir="auto"><h1 tabindex="-1" class="heading-element" dir="auto" _msttexthash="11422697" _msthash="317">我们走吧</h1><a id="user-content-lets-fucking-go" class="anchor" aria-label="永久链接：我们他妈的走吧" href="#lets-fucking-go" _mstaria-label="563849" _msthash="318"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></div>
<div dir="auto">
    <a target="_blank" rel="noopener noreferrer" href="/naklecha/llama3-from-scratch/blob/main/images/42.png"><img src="/naklecha/llama3-from-scratch/raw/main/images/42.png" width="600px" style="max-width: 100%;"></a>
</div>
<div class="highlight highlight-source-python notranslate position-relative overflow-auto" dir="auto"><pre><span class="pl-s1">tokenizer</span>.<span class="pl-en">decode</span>([<span class="pl-s1">next_token</span>.<span class="pl-en">item</span>()])</pre><div class="zeroclipboard-container">
    <clipboard-copy aria-label="Copy" class="ClipboardButton btn btn-invisible js-clipboard-copy m-2 p-0 d-flex flex-justify-center flex-items-center" data-copy-feedback="Copied!" data-tooltip-direction="w" value="tokenizer.decode([next_token.item()])" tabindex="0" role="button">
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-copy js-clipboard-copy-icon">
    <path d="M0 6.75C0 5.784.784 5 1.75 5h1.5a.75.75 0 0 1 0 1.5h-1.5a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-1.5a.75.75 0 0 1 1.5 0v1.5A1.75 1.75 0 0 1 9.25 16h-7.5A1.75 1.75 0 0 1 0 14.25Z"></path><path d="M5 1.75C5 .784 5.784 0 6.75 0h7.5C15.216 0 16 .784 16 1.75v7.5A1.75 1.75 0 0 1 14.25 11h-7.5A1.75 1.75 0 0 1 5 9.25Zm1.75-.25a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-7.5a.25.25 0 0 0-.25-.25Z"></path>
</svg>
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-check js-clipboard-check-icon color-fg-success d-none">
    <path d="M13.78 4.22a.75.75 0 0 1 0 1.06l-7.25 7.25a.75.75 0 0 1-1.06 0L2.22 9.28a.751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018L6 10.94l6.72-6.72a.75.75 0 0 1 1.06 0Z"></path>
</svg>
    </clipboard-copy>
  </div></div>
<div class="snippet-clipboard-content notranslate position-relative overflow-auto"><pre class="notranslate"><code>'42'
</code></pre><div class="zeroclipboard-container">
    <clipboard-copy aria-label="Copy" class="ClipboardButton btn btn-invisible js-clipboard-copy m-2 p-0 d-flex flex-justify-center flex-items-center" data-copy-feedback="Copied!" data-tooltip-direction="w" value="'42'" tabindex="0" role="button">
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-copy js-clipboard-copy-icon">
    <path d="M0 6.75C0 5.784.784 5 1.75 5h1.5a.75.75 0 0 1 0 1.5h-1.5a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-1.5a.75.75 0 0 1 1.5 0v1.5A1.75 1.75 0 0 1 9.25 16h-7.5A1.75 1.75 0 0 1 0 14.25Z"></path><path d="M5 1.75C5 .784 5.784 0 6.75 0h7.5C15.216 0 16 .784 16 1.75v7.5A1.75 1.75 0 0 1 14.25 11h-7.5A1.75 1.75 0 0 1 5 9.25Zm1.75-.25a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-7.5a.25.25 0 0 0-.25-.25Z"></path>
</svg>
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-check js-clipboard-check-icon color-fg-success d-none">
    <path d="M13.78 4.22a.75.75 0 0 1 0 1.06l-7.25 7.25a.75.75 0 0 1-1.06 0L2.22 9.28a.751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018L6 10.94l6.72-6.72a.75.75 0 0 1 1.06 0Z"></path>
</svg>
    </clipboard-copy>
  </div></div>
<div class="markdown-heading" dir="auto"><h1 tabindex="-1" class="heading-element" dir="auto" _msttexthash="29463824" _msthash="319">谢谢你，我爱你:)</h1><a id="user-content-thank-you-i-love-you-" class="anchor" aria-label="永久链接：谢谢你，我爱你:)" href="#thank-you-i-love-you-" _mstaria-label="784147" _msthash="320"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></div>
<p dir="auto" _msttexthash="84856577" _msthash="321">这就是结束。希望您喜欢阅读它！</p>
<p dir="auto" _msttexthash="37300705" _msthash="322">如果您想支持我的工作</p>
<ol dir="auto">
<li _msttexthash="23564801" _msthash="323">在 Twitter 上关注我 <a href="https://twitter.com/naklecha" rel="nofollow" _istranslated="1">https://twitter.com/naklecha</a></li>
<li _msttexthash="38552254" _msthash="324">或者，给我买杯咖啡 <a href="https://www.buymeacoffee.com/naklecha" rel="nofollow" _istranslated="1">https://www.buymeacoffee.com/naklecha</a></li>
</ol>
<p dir="auto" _msttexthash="147456959" _msthash="325">老实说，如果你走到了这一步，你已经让我的一天:)</p>
<div class="markdown-heading" dir="auto"><h2 tabindex="-1" class="heading-element" dir="auto" _msttexthash="34459711" _msthash="326">是什么激励着我？</h2><a id="user-content-what-motivates-me" class="anchor" aria-label="永久链接：是什么激励着我？" href="#what-motivates-me" _mstaria-label="681798" _msthash="327"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></div>
<p dir="auto" _msttexthash="356709535" _msthash="328">我和我的朋友们肩负着一个使命 - 让研究更容易获得！
我们创建了一个名为 A10 的研究实验室 - <a href="http://aaaaaaaaaa.org/" rel="nofollow" _istranslated="1">AAAAAAAAAA.org</a></p>
<p dir="auto" _msttexthash="8744606" _msthash="329">A10 推特 - <a href="https://twitter.com/aaaaaaaaaaorg" rel="nofollow" _istranslated="1">https://twitter.com/aaaaaaaaaaorg</a></p>
<p dir="auto" _msttexthash="26490100" _msthash="330">我们的论文：</p>
<div dir="auto">
    <a target="_blank" rel="noopener noreferrer" href="/naklecha/llama3-from-scratch/blob/main/images/a10.png"><img src="/naklecha/llama3-from-scratch/raw/main/images/a10.png" width="600px" style="max-width: 100%;"></a>
</div>
</article></div>
