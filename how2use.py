import eval_utils as utils

# tokens has to be seperated by spaces.

# 指令预处理在Normalize.py，指令之间以<eos>隔开，最长处理512
text="add al ah <eos> inc edi <eos> dec esi <eos> add [eax] al <eos> pad-normal <eos> add al dh <eos> inc edi <eos> dec esi <eos> add [eax] al <eos> pad-normal <eos> add al ah <eos> dec esp <eos> dec esi <eos> add [eax] al <eos> pad-normal <eos> add al dh <eos> dec ebp <eos> dec esi <eos> add [eax] al <eos> pad-normal <eos> add [ecx] al <eos> pad-normal <eos> add [esi_const-abnormal] dl <eos> pop esp <eos> dec esi <eos> add [eax] al <eos> pad-normal <eos> add al dh <eos> pop esp <eos> dec esi <eos> add [eax] al <eos> pad-normal <eos> add [eax] dh <eos> pop ebp <eos> dec esi <eos> add [eax] al <eos> pad-normal <eos> add [eax_const-normal] dl <eos> dec esi <eos> add [eax] al <eos> pad-normal <eos> add [eax_const-normal] dh <eos> dec esi <eos> add [eax] al <eos> pad-normal <eos> add [eax] dl <eos> xchg ebp eax <eos> dec esi <eos> add [eax] al <eos> pad-normal <eos> add [ebp_const-normal] dl <eos> pad-normal <eos> add [ebx] bl <eos> dec esi <eos> add [eax] al <eos> pad-normal <eos> add [eax] ah <eos> cwde <eos> dec esi <eos> add [eax] al <eos> pad-normal <eos> add [eax] ah <eos> xchg esp eax <eos> dec esi <eos> add [eax] al <eos> pad-normal <eos> add [eax_const-normal] dl <eos> pad-normal <eos> add al dl <eos> xchg esp eax <eos> dec esi <eos> add [eax] al <eos> pad-normal <eos> add al al <eos> cmpsd [esi] es [edi] <eos> dec esi <eos> add [eax] al <eos> pad-normal <eos> add [eax_const-normal] ah <eos> dec esi <eos> add [eax] al <eos> pad-normal <eos> add [eax_const-normal] al <eos> pad-normal <eos> add [eax_const-normal] dl <eos> dec esi <eos> add [eax] al <eos> pad-normal <eos> add [eax] ah <eos> stosb es [edi] al <eos> dec esi <eos> add [eax] al <eos> pad-normal <eos> add [esi_ecx*const] bl <eos> pad-normal <eos> add al al <eos> lodsd eax [esi] <eos> dec esi <eos> add [eax] al <eos> pad-normal <eos> add [eax_const-normal] dh <eos> dec esi <eos> add [eax] al <eos> pad-normal <eos> add al ah <eos> mov bl const <eos> pad-normal <eos> add al dl <eos> xchg edi eax <eos> dec esi <eos> add [eax] al <eos> pad-normal <eos> add [eax] dh <eos> mov ah const <eos> pad-normal <eos> add [eax_const-normal] ah <eos> pad-normal <eos> add [eax] ah <eos> mov bh const <eos> pad-normal <eos> add [eax_const-normal] dl <eos> dec esi <eos> add [eax] al <eos> pad-normal <eos> add [eax_const-normal] ah <eos> dec esi <eos> add [eax] al <eos> pad-normal <eos> add [eax_const-normal] dh <eos> pad-normal <eos> add [eax_const-normal] al <eos> dec esi <eos> add [eax] al <eos> pad-normal <eos> add [eax] dl <eos> pad-abnormal <eos> dec esi <eos> add [eax] al <eos> pad-normal <eos> add [edx_const-normal] ah <eos> pad-normal <eos> add [eax_const-normal] dl <eos> dec esi <eos> add [eax] al <eos> pad-normal <eos> add [eax_const-normal] dh <eos> dec ebp <eos> add [eax] al <eos> pad-normal <eos> add [eax_const-normal] ah <eos> dec ebp <eos> add [eax] al <eos> pad-normal <eos> add [eax_const-normal] dl <eos> dec ebp <eos> add [eax] al <eos> pad-normal <eos>"
packeralm = utils.UsableTransformer()

embeddings = packeralm.encode(text)

print("usable embedding of this basicblock:", embeddings)
print("the shape of output tensor: ", embeddings.shape)