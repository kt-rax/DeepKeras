??
??
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape?"serve*2.1.02v2.1.0-rc2-17-ge5bf8de8??
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
?
conv2d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@* 
shared_nameconv2d_3/kernel
{
#conv2d_3/kernel/Read/ReadVariableOpReadVariableOpconv2d_3/kernel*&
_output_shapes
:@*
dtype0
r
conv2d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_3/bias
k
!conv2d_3/bias/Read/ReadVariableOpReadVariableOpconv2d_3/bias*
_output_shapes
:@*
dtype0
?
conv2d_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?* 
shared_nameconv2d_4/kernel
|
#conv2d_4/kernel/Read/ReadVariableOpReadVariableOpconv2d_4/kernel*'
_output_shapes
:@?*
dtype0
s
conv2d_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameconv2d_4/bias
l
!conv2d_4/bias/Read/ReadVariableOpReadVariableOpconv2d_4/bias*
_output_shapes	
:?*
dtype0
?
conv2d_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??* 
shared_nameconv2d_5/kernel
}
#conv2d_5/kernel/Read/ReadVariableOpReadVariableOpconv2d_5/kernel*(
_output_shapes
:??*
dtype0
s
conv2d_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameconv2d_5/bias
l
!conv2d_5/bias/Read/ReadVariableOpReadVariableOpconv2d_5/bias*
_output_shapes	
:?*
dtype0
?
latent_vector_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
? ?*'
shared_namelatent_vector_1/kernel
?
*latent_vector_1/kernel/Read/ReadVariableOpReadVariableOplatent_vector_1/kernel* 
_output_shapes
:
? ?*
dtype0
?
latent_vector_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_namelatent_vector_1/bias
z
(latent_vector_1/bias/Read/ReadVariableOpReadVariableOplatent_vector_1/bias*
_output_shapes	
:?*
dtype0
z
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
?? *
shared_namedense_1/kernel
s
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel* 
_output_shapes
:
?? *
dtype0
q
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:? *
shared_namedense_1/bias
j
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes	
:? *
dtype0
?
conv2d_transpose_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:?**
shared_nameconv2d_transpose_3/kernel
?
-conv2d_transpose_3/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_3/kernel*'
_output_shapes
:?*
dtype0
?
conv2d_transpose_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameconv2d_transpose_3/bias

+conv2d_transpose_3/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_3/bias*
_output_shapes
:*
dtype0
?
conv2d_transpose_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameconv2d_transpose_4/kernel
?
-conv2d_transpose_4/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_4/kernel*&
_output_shapes
:*
dtype0
?
conv2d_transpose_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameconv2d_transpose_4/bias

+conv2d_transpose_4/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_4/bias*
_output_shapes
:*
dtype0
?
conv2d_transpose_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameconv2d_transpose_5/kernel
?
-conv2d_transpose_5/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_5/kernel*&
_output_shapes
:*
dtype0
?
conv2d_transpose_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameconv2d_transpose_5/bias

+conv2d_transpose_5/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_5/bias*
_output_shapes
:*
dtype0
?
decoder_output_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_namedecoder_output_1/kernel
?
+decoder_output_1/kernel/Read/ReadVariableOpReadVariableOpdecoder_output_1/kernel*&
_output_shapes
:*
dtype0
?
decoder_output_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_namedecoder_output_1/bias
{
)decoder_output_1/bias/Read/ReadVariableOpReadVariableOpdecoder_output_1/bias*
_output_shapes
:*
dtype0
?
Adam/conv2d_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_nameAdam/conv2d_3/kernel/m
?
*Adam/conv2d_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_3/kernel/m*&
_output_shapes
:@*
dtype0
?
Adam/conv2d_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv2d_3/bias/m
y
(Adam/conv2d_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_3/bias/m*
_output_shapes
:@*
dtype0
?
Adam/conv2d_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*'
shared_nameAdam/conv2d_4/kernel/m
?
*Adam/conv2d_4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_4/kernel/m*'
_output_shapes
:@?*
dtype0
?
Adam/conv2d_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/conv2d_4/bias/m
z
(Adam/conv2d_4/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_4/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/conv2d_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*'
shared_nameAdam/conv2d_5/kernel/m
?
*Adam/conv2d_5/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_5/kernel/m*(
_output_shapes
:??*
dtype0
?
Adam/conv2d_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/conv2d_5/bias/m
z
(Adam/conv2d_5/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_5/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/latent_vector_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
? ?*.
shared_nameAdam/latent_vector_1/kernel/m
?
1Adam/latent_vector_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/latent_vector_1/kernel/m* 
_output_shapes
:
? ?*
dtype0
?
Adam/latent_vector_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*,
shared_nameAdam/latent_vector_1/bias/m
?
/Adam/latent_vector_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/latent_vector_1/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
?? *&
shared_nameAdam/dense_1/kernel/m
?
)Adam/dense_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/m* 
_output_shapes
:
?? *
dtype0

Adam/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:? *$
shared_nameAdam/dense_1/bias/m
x
'Adam/dense_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/m*
_output_shapes	
:? *
dtype0
?
 Adam/conv2d_transpose_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*1
shared_name" Adam/conv2d_transpose_3/kernel/m
?
4Adam/conv2d_transpose_3/kernel/m/Read/ReadVariableOpReadVariableOp Adam/conv2d_transpose_3/kernel/m*'
_output_shapes
:?*
dtype0
?
Adam/conv2d_transpose_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name Adam/conv2d_transpose_3/bias/m
?
2Adam/conv2d_transpose_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_3/bias/m*
_output_shapes
:*
dtype0
?
 Adam/conv2d_transpose_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/conv2d_transpose_4/kernel/m
?
4Adam/conv2d_transpose_4/kernel/m/Read/ReadVariableOpReadVariableOp Adam/conv2d_transpose_4/kernel/m*&
_output_shapes
:*
dtype0
?
Adam/conv2d_transpose_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name Adam/conv2d_transpose_4/bias/m
?
2Adam/conv2d_transpose_4/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_4/bias/m*
_output_shapes
:*
dtype0
?
 Adam/conv2d_transpose_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/conv2d_transpose_5/kernel/m
?
4Adam/conv2d_transpose_5/kernel/m/Read/ReadVariableOpReadVariableOp Adam/conv2d_transpose_5/kernel/m*&
_output_shapes
:*
dtype0
?
Adam/conv2d_transpose_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name Adam/conv2d_transpose_5/bias/m
?
2Adam/conv2d_transpose_5/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_5/bias/m*
_output_shapes
:*
dtype0
?
Adam/decoder_output_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name Adam/decoder_output_1/kernel/m
?
2Adam/decoder_output_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/decoder_output_1/kernel/m*&
_output_shapes
:*
dtype0
?
Adam/decoder_output_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_nameAdam/decoder_output_1/bias/m
?
0Adam/decoder_output_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/decoder_output_1/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv2d_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_nameAdam/conv2d_3/kernel/v
?
*Adam/conv2d_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_3/kernel/v*&
_output_shapes
:@*
dtype0
?
Adam/conv2d_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv2d_3/bias/v
y
(Adam/conv2d_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_3/bias/v*
_output_shapes
:@*
dtype0
?
Adam/conv2d_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*'
shared_nameAdam/conv2d_4/kernel/v
?
*Adam/conv2d_4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_4/kernel/v*'
_output_shapes
:@?*
dtype0
?
Adam/conv2d_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/conv2d_4/bias/v
z
(Adam/conv2d_4/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_4/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/conv2d_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*'
shared_nameAdam/conv2d_5/kernel/v
?
*Adam/conv2d_5/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_5/kernel/v*(
_output_shapes
:??*
dtype0
?
Adam/conv2d_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/conv2d_5/bias/v
z
(Adam/conv2d_5/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_5/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/latent_vector_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
? ?*.
shared_nameAdam/latent_vector_1/kernel/v
?
1Adam/latent_vector_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/latent_vector_1/kernel/v* 
_output_shapes
:
? ?*
dtype0
?
Adam/latent_vector_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*,
shared_nameAdam/latent_vector_1/bias/v
?
/Adam/latent_vector_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/latent_vector_1/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
?? *&
shared_nameAdam/dense_1/kernel/v
?
)Adam/dense_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/v* 
_output_shapes
:
?? *
dtype0

Adam/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:? *$
shared_nameAdam/dense_1/bias/v
x
'Adam/dense_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/v*
_output_shapes	
:? *
dtype0
?
 Adam/conv2d_transpose_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*1
shared_name" Adam/conv2d_transpose_3/kernel/v
?
4Adam/conv2d_transpose_3/kernel/v/Read/ReadVariableOpReadVariableOp Adam/conv2d_transpose_3/kernel/v*'
_output_shapes
:?*
dtype0
?
Adam/conv2d_transpose_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name Adam/conv2d_transpose_3/bias/v
?
2Adam/conv2d_transpose_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_3/bias/v*
_output_shapes
:*
dtype0
?
 Adam/conv2d_transpose_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/conv2d_transpose_4/kernel/v
?
4Adam/conv2d_transpose_4/kernel/v/Read/ReadVariableOpReadVariableOp Adam/conv2d_transpose_4/kernel/v*&
_output_shapes
:*
dtype0
?
Adam/conv2d_transpose_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name Adam/conv2d_transpose_4/bias/v
?
2Adam/conv2d_transpose_4/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_4/bias/v*
_output_shapes
:*
dtype0
?
 Adam/conv2d_transpose_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/conv2d_transpose_5/kernel/v
?
4Adam/conv2d_transpose_5/kernel/v/Read/ReadVariableOpReadVariableOp Adam/conv2d_transpose_5/kernel/v*&
_output_shapes
:*
dtype0
?
Adam/conv2d_transpose_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name Adam/conv2d_transpose_5/bias/v
?
2Adam/conv2d_transpose_5/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_5/bias/v*
_output_shapes
:*
dtype0
?
Adam/decoder_output_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name Adam/decoder_output_1/kernel/v
?
2Adam/decoder_output_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/decoder_output_1/kernel/v*&
_output_shapes
:*
dtype0
?
Adam/decoder_output_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_nameAdam/decoder_output_1/bias/v
?
0Adam/decoder_output_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/decoder_output_1/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
?\
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?[
value?[B?[ B?[
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
	optimizer
regularization_losses
	variables
trainable_variables
	keras_api
	
signatures
 
?
layer-0

layer_with_weights-0

layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer-4
layer_with_weights-3
layer-5
regularization_losses
	variables
trainable_variables
	keras_api
?
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer_with_weights-4
layer-6
regularization_losses
	variables
trainable_variables
	keras_api
?
iter

beta_1

 beta_2
	!decay
"learning_rate#m?$m?%m?&m?'m?(m?)m?*m?+m?,m?-m?.m?/m?0m?1m?2m?3m?4m?#v?$v?%v?&v?'v?(v?)v?*v?+v?,v?-v?.v?/v?0v?1v?2v?3v?4v?
 
?
#0
$1
%2
&3
'4
(5
)6
*7
+8
,9
-10
.11
/12
013
114
215
316
417
?
#0
$1
%2
&3
'4
(5
)6
*7
+8
,9
-10
.11
/12
013
114
215
316
417
?
5metrics
regularization_losses
	variables
6layer_regularization_losses
7non_trainable_variables
trainable_variables

8layers
 
h

#kernel
$bias
9regularization_losses
:	variables
;trainable_variables
<	keras_api
h

%kernel
&bias
=regularization_losses
>	variables
?trainable_variables
@	keras_api
h

'kernel
(bias
Aregularization_losses
B	variables
Ctrainable_variables
D	keras_api
R
Eregularization_losses
F	variables
Gtrainable_variables
H	keras_api
h

)kernel
*bias
Iregularization_losses
J	variables
Ktrainable_variables
L	keras_api
 
8
#0
$1
%2
&3
'4
(5
)6
*7
8
#0
$1
%2
&3
'4
(5
)6
*7
?
Mmetrics
regularization_losses
	variables
Nlayer_regularization_losses
Onon_trainable_variables
trainable_variables

Players
 
h

+kernel
,bias
Qregularization_losses
R	variables
Strainable_variables
T	keras_api
R
Uregularization_losses
V	variables
Wtrainable_variables
X	keras_api
h

-kernel
.bias
Yregularization_losses
Z	variables
[trainable_variables
\	keras_api
h

/kernel
0bias
]regularization_losses
^	variables
_trainable_variables
`	keras_api
h

1kernel
2bias
aregularization_losses
b	variables
ctrainable_variables
d	keras_api
h

3kernel
4bias
eregularization_losses
f	variables
gtrainable_variables
h	keras_api
 
F
+0
,1
-2
.3
/4
05
16
27
38
49
F
+0
,1
-2
.3
/4
05
16
27
38
49
?
imetrics
regularization_losses
	variables
jlayer_regularization_losses
knon_trainable_variables
trainable_variables

llayers
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEconv2d_3/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEconv2d_3/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEconv2d_4/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEconv2d_4/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEconv2d_5/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEconv2d_5/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUElatent_vector_1/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUElatent_vector_1/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEdense_1/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE
HF
VARIABLE_VALUEdense_1/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEconv2d_transpose_3/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEconv2d_transpose_3/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEconv2d_transpose_4/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEconv2d_transpose_4/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEconv2d_transpose_5/kernel'variables/14/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEconv2d_transpose_5/bias'variables/15/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEdecoder_output_1/kernel'variables/16/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEdecoder_output_1/bias'variables/17/.ATTRIBUTES/VARIABLE_VALUE
 
 
 

0
1
2
 

#0
$1

#0
$1
?
mmetrics
9regularization_losses
:	variables
nlayer_regularization_losses
onon_trainable_variables
;trainable_variables

players
 

%0
&1

%0
&1
?
qmetrics
=regularization_losses
>	variables
rlayer_regularization_losses
snon_trainable_variables
?trainable_variables

tlayers
 

'0
(1

'0
(1
?
umetrics
Aregularization_losses
B	variables
vlayer_regularization_losses
wnon_trainable_variables
Ctrainable_variables

xlayers
 
 
 
?
ymetrics
Eregularization_losses
F	variables
zlayer_regularization_losses
{non_trainable_variables
Gtrainable_variables

|layers
 

)0
*1

)0
*1
?
}metrics
Iregularization_losses
J	variables
~layer_regularization_losses
non_trainable_variables
Ktrainable_variables
?layers
 
 
 
*
0

1
2
3
4
5
 

+0
,1

+0
,1
?
?metrics
Qregularization_losses
R	variables
 ?layer_regularization_losses
?non_trainable_variables
Strainable_variables
?layers
 
 
 
?
?metrics
Uregularization_losses
V	variables
 ?layer_regularization_losses
?non_trainable_variables
Wtrainable_variables
?layers
 

-0
.1

-0
.1
?
?metrics
Yregularization_losses
Z	variables
 ?layer_regularization_losses
?non_trainable_variables
[trainable_variables
?layers
 

/0
01

/0
01
?
?metrics
]regularization_losses
^	variables
 ?layer_regularization_losses
?non_trainable_variables
_trainable_variables
?layers
 

10
21

10
21
?
?metrics
aregularization_losses
b	variables
 ?layer_regularization_losses
?non_trainable_variables
ctrainable_variables
?layers
 

30
41

30
41
?
?metrics
eregularization_losses
f	variables
 ?layer_regularization_losses
?non_trainable_variables
gtrainable_variables
?layers
 
 
 
1
0
1
2
3
4
5
6
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
nl
VARIABLE_VALUEAdam/conv2d_3/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/conv2d_3/bias/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/conv2d_4/kernel/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/conv2d_4/bias/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/conv2d_5/kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/conv2d_5/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/latent_vector_1/kernel/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUEAdam/latent_vector_1/bias/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_1/kernel/mBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUEAdam/dense_1/bias/mBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE Adam/conv2d_transpose_3/kernel/mCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/conv2d_transpose_3/bias/mCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE Adam/conv2d_transpose_4/kernel/mCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/conv2d_transpose_4/bias/mCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE Adam/conv2d_transpose_5/kernel/mCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/conv2d_transpose_5/bias/mCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/decoder_output_1/kernel/mCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/decoder_output_1/bias/mCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/conv2d_3/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/conv2d_3/bias/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/conv2d_4/kernel/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/conv2d_4/bias/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/conv2d_5/kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/conv2d_5/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/latent_vector_1/kernel/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUEAdam/latent_vector_1/bias/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_1/kernel/vBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUEAdam/dense_1/bias/vBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE Adam/conv2d_transpose_3/kernel/vCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/conv2d_transpose_3/bias/vCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE Adam/conv2d_transpose_4/kernel/vCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/conv2d_transpose_4/bias/vCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE Adam/conv2d_transpose_5/kernel/vCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/conv2d_transpose_5/bias/vCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/decoder_output_1/kernel/vCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/decoder_output_1/bias/vCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_encoder_inputPlaceholder*/
_output_shapes
:?????????  *
dtype0*$
shape:?????????  
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_encoder_inputconv2d_3/kernelconv2d_3/biasconv2d_4/kernelconv2d_4/biasconv2d_5/kernelconv2d_5/biaslatent_vector_1/kernellatent_vector_1/biasdense_1/kerneldense_1/biasconv2d_transpose_3/kernelconv2d_transpose_3/biasconv2d_transpose_4/kernelconv2d_transpose_4/biasconv2d_transpose_5/kernelconv2d_transpose_5/biasdecoder_output_1/kerneldecoder_output_1/bias*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:?????????  *-
config_proto

GPU

CPU2*0J 8*,
f'R%
#__inference_signature_wrapper_88454
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp#conv2d_3/kernel/Read/ReadVariableOp!conv2d_3/bias/Read/ReadVariableOp#conv2d_4/kernel/Read/ReadVariableOp!conv2d_4/bias/Read/ReadVariableOp#conv2d_5/kernel/Read/ReadVariableOp!conv2d_5/bias/Read/ReadVariableOp*latent_vector_1/kernel/Read/ReadVariableOp(latent_vector_1/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp-conv2d_transpose_3/kernel/Read/ReadVariableOp+conv2d_transpose_3/bias/Read/ReadVariableOp-conv2d_transpose_4/kernel/Read/ReadVariableOp+conv2d_transpose_4/bias/Read/ReadVariableOp-conv2d_transpose_5/kernel/Read/ReadVariableOp+conv2d_transpose_5/bias/Read/ReadVariableOp+decoder_output_1/kernel/Read/ReadVariableOp)decoder_output_1/bias/Read/ReadVariableOp*Adam/conv2d_3/kernel/m/Read/ReadVariableOp(Adam/conv2d_3/bias/m/Read/ReadVariableOp*Adam/conv2d_4/kernel/m/Read/ReadVariableOp(Adam/conv2d_4/bias/m/Read/ReadVariableOp*Adam/conv2d_5/kernel/m/Read/ReadVariableOp(Adam/conv2d_5/bias/m/Read/ReadVariableOp1Adam/latent_vector_1/kernel/m/Read/ReadVariableOp/Adam/latent_vector_1/bias/m/Read/ReadVariableOp)Adam/dense_1/kernel/m/Read/ReadVariableOp'Adam/dense_1/bias/m/Read/ReadVariableOp4Adam/conv2d_transpose_3/kernel/m/Read/ReadVariableOp2Adam/conv2d_transpose_3/bias/m/Read/ReadVariableOp4Adam/conv2d_transpose_4/kernel/m/Read/ReadVariableOp2Adam/conv2d_transpose_4/bias/m/Read/ReadVariableOp4Adam/conv2d_transpose_5/kernel/m/Read/ReadVariableOp2Adam/conv2d_transpose_5/bias/m/Read/ReadVariableOp2Adam/decoder_output_1/kernel/m/Read/ReadVariableOp0Adam/decoder_output_1/bias/m/Read/ReadVariableOp*Adam/conv2d_3/kernel/v/Read/ReadVariableOp(Adam/conv2d_3/bias/v/Read/ReadVariableOp*Adam/conv2d_4/kernel/v/Read/ReadVariableOp(Adam/conv2d_4/bias/v/Read/ReadVariableOp*Adam/conv2d_5/kernel/v/Read/ReadVariableOp(Adam/conv2d_5/bias/v/Read/ReadVariableOp1Adam/latent_vector_1/kernel/v/Read/ReadVariableOp/Adam/latent_vector_1/bias/v/Read/ReadVariableOp)Adam/dense_1/kernel/v/Read/ReadVariableOp'Adam/dense_1/bias/v/Read/ReadVariableOp4Adam/conv2d_transpose_3/kernel/v/Read/ReadVariableOp2Adam/conv2d_transpose_3/bias/v/Read/ReadVariableOp4Adam/conv2d_transpose_4/kernel/v/Read/ReadVariableOp2Adam/conv2d_transpose_4/bias/v/Read/ReadVariableOp4Adam/conv2d_transpose_5/kernel/v/Read/ReadVariableOp2Adam/conv2d_transpose_5/bias/v/Read/ReadVariableOp2Adam/decoder_output_1/kernel/v/Read/ReadVariableOp0Adam/decoder_output_1/bias/v/Read/ReadVariableOpConst*H
TinA
?2=	*
Tout
2*,
_gradient_op_typePartitionedCallUnused*
_output_shapes
: *-
config_proto

GPU

CPU2*0J 8*'
f"R 
__inference__traced_save_89505
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rateconv2d_3/kernelconv2d_3/biasconv2d_4/kernelconv2d_4/biasconv2d_5/kernelconv2d_5/biaslatent_vector_1/kernellatent_vector_1/biasdense_1/kerneldense_1/biasconv2d_transpose_3/kernelconv2d_transpose_3/biasconv2d_transpose_4/kernelconv2d_transpose_4/biasconv2d_transpose_5/kernelconv2d_transpose_5/biasdecoder_output_1/kerneldecoder_output_1/biasAdam/conv2d_3/kernel/mAdam/conv2d_3/bias/mAdam/conv2d_4/kernel/mAdam/conv2d_4/bias/mAdam/conv2d_5/kernel/mAdam/conv2d_5/bias/mAdam/latent_vector_1/kernel/mAdam/latent_vector_1/bias/mAdam/dense_1/kernel/mAdam/dense_1/bias/m Adam/conv2d_transpose_3/kernel/mAdam/conv2d_transpose_3/bias/m Adam/conv2d_transpose_4/kernel/mAdam/conv2d_transpose_4/bias/m Adam/conv2d_transpose_5/kernel/mAdam/conv2d_transpose_5/bias/mAdam/decoder_output_1/kernel/mAdam/decoder_output_1/bias/mAdam/conv2d_3/kernel/vAdam/conv2d_3/bias/vAdam/conv2d_4/kernel/vAdam/conv2d_4/bias/vAdam/conv2d_5/kernel/vAdam/conv2d_5/bias/vAdam/latent_vector_1/kernel/vAdam/latent_vector_1/bias/vAdam/dense_1/kernel/vAdam/dense_1/bias/v Adam/conv2d_transpose_3/kernel/vAdam/conv2d_transpose_3/bias/v Adam/conv2d_transpose_4/kernel/vAdam/conv2d_transpose_4/bias/v Adam/conv2d_transpose_5/kernel/vAdam/conv2d_transpose_5/bias/vAdam/decoder_output_1/kernel/vAdam/decoder_output_1/bias/v*G
Tin@
>2<*
Tout
2*,
_gradient_op_typePartitionedCallUnused*
_output_shapes
: *-
config_proto

GPU

CPU2*0J 8**
f%R#
!__inference__traced_restore_89694??
?
?
(__inference_conv2d_3_layer_call_fn_87728

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+???????????????????????????@*-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_conv2d_3_layer_call_and_return_conditional_losses_877202
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?$
?
M__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_88019

inputs,
(conv2d_transpose_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/3?
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
2
conv2d_transpose?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:& "
 
_user_specified_nameinputs
?
?
H__inference_latent_vector_layer_call_and_return_conditional_losses_89261

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
? ?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
?
E
)__inference_reshape_1_layer_call_fn_89304

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:??????????*-
config_proto

GPU

CPU2*0J 8*M
fHRF
D__inference_reshape_1_layer_call_and_return_conditional_losses_881102
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:?????????? :& "
 
_user_specified_nameinputs
?
?
2__inference_conv2d_transpose_3_layer_call_fn_87941

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+???????????????????????????*-
config_proto

GPU

CPU2*0J 8*V
fQRO
M__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_879332
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,????????????????????????????::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?
?
C__inference_conv2d_5_layer_call_and_return_conditional_losses_87762

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOpo
dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
dilation_rate?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????2	
BiasAdds
ReluReluBiasAdd:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,????????????????????????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:& "
 
_user_specified_nameinputs
??
?
F__inference_autoencoder_layer_call_and_return_conditional_losses_88792

inputs3
/encoder_conv2d_3_conv2d_readvariableop_resource4
0encoder_conv2d_3_biasadd_readvariableop_resource3
/encoder_conv2d_4_conv2d_readvariableop_resource4
0encoder_conv2d_4_biasadd_readvariableop_resource3
/encoder_conv2d_5_conv2d_readvariableop_resource4
0encoder_conv2d_5_biasadd_readvariableop_resource8
4encoder_latent_vector_matmul_readvariableop_resource9
5encoder_latent_vector_biasadd_readvariableop_resource2
.decoder_dense_1_matmul_readvariableop_resource3
/decoder_dense_1_biasadd_readvariableop_resourceG
Cdecoder_conv2d_transpose_3_conv2d_transpose_readvariableop_resource>
:decoder_conv2d_transpose_3_biasadd_readvariableop_resourceG
Cdecoder_conv2d_transpose_4_conv2d_transpose_readvariableop_resource>
:decoder_conv2d_transpose_4_biasadd_readvariableop_resourceG
Cdecoder_conv2d_transpose_5_conv2d_transpose_readvariableop_resource>
:decoder_conv2d_transpose_5_biasadd_readvariableop_resourceC
?decoder_decoder_output_conv2d_transpose_readvariableop_resource:
6decoder_decoder_output_biasadd_readvariableop_resource
identity??1decoder/conv2d_transpose_3/BiasAdd/ReadVariableOp?:decoder/conv2d_transpose_3/conv2d_transpose/ReadVariableOp?1decoder/conv2d_transpose_4/BiasAdd/ReadVariableOp?:decoder/conv2d_transpose_4/conv2d_transpose/ReadVariableOp?1decoder/conv2d_transpose_5/BiasAdd/ReadVariableOp?:decoder/conv2d_transpose_5/conv2d_transpose/ReadVariableOp?-decoder/decoder_output/BiasAdd/ReadVariableOp?6decoder/decoder_output/conv2d_transpose/ReadVariableOp?&decoder/dense_1/BiasAdd/ReadVariableOp?%decoder/dense_1/MatMul/ReadVariableOp?'encoder/conv2d_3/BiasAdd/ReadVariableOp?&encoder/conv2d_3/Conv2D/ReadVariableOp?'encoder/conv2d_4/BiasAdd/ReadVariableOp?&encoder/conv2d_4/Conv2D/ReadVariableOp?'encoder/conv2d_5/BiasAdd/ReadVariableOp?&encoder/conv2d_5/Conv2D/ReadVariableOp?,encoder/latent_vector/BiasAdd/ReadVariableOp?+encoder/latent_vector/MatMul/ReadVariableOp?
&encoder/conv2d_3/Conv2D/ReadVariableOpReadVariableOp/encoder_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02(
&encoder/conv2d_3/Conv2D/ReadVariableOp?
encoder/conv2d_3/Conv2DConv2Dinputs.encoder/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
encoder/conv2d_3/Conv2D?
'encoder/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp0encoder_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02)
'encoder/conv2d_3/BiasAdd/ReadVariableOp?
encoder/conv2d_3/BiasAddBiasAdd encoder/conv2d_3/Conv2D:output:0/encoder/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
encoder/conv2d_3/BiasAdd?
encoder/conv2d_3/ReluRelu!encoder/conv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
encoder/conv2d_3/Relu?
&encoder/conv2d_4/Conv2D/ReadVariableOpReadVariableOp/encoder_conv2d_4_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02(
&encoder/conv2d_4/Conv2D/ReadVariableOp?
encoder/conv2d_4/Conv2DConv2D#encoder/conv2d_3/Relu:activations:0.encoder/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
encoder/conv2d_4/Conv2D?
'encoder/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp0encoder_conv2d_4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02)
'encoder/conv2d_4/BiasAdd/ReadVariableOp?
encoder/conv2d_4/BiasAddBiasAdd encoder/conv2d_4/Conv2D:output:0/encoder/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
encoder/conv2d_4/BiasAdd?
encoder/conv2d_4/ReluRelu!encoder/conv2d_4/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
encoder/conv2d_4/Relu?
&encoder/conv2d_5/Conv2D/ReadVariableOpReadVariableOp/encoder_conv2d_5_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02(
&encoder/conv2d_5/Conv2D/ReadVariableOp?
encoder/conv2d_5/Conv2DConv2D#encoder/conv2d_4/Relu:activations:0.encoder/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
encoder/conv2d_5/Conv2D?
'encoder/conv2d_5/BiasAdd/ReadVariableOpReadVariableOp0encoder_conv2d_5_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02)
'encoder/conv2d_5/BiasAdd/ReadVariableOp?
encoder/conv2d_5/BiasAddBiasAdd encoder/conv2d_5/Conv2D:output:0/encoder/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
encoder/conv2d_5/BiasAdd?
encoder/conv2d_5/ReluRelu!encoder/conv2d_5/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
encoder/conv2d_5/Relu?
encoder/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
encoder/flatten_1/Const?
encoder/flatten_1/ReshapeReshape#encoder/conv2d_5/Relu:activations:0 encoder/flatten_1/Const:output:0*
T0*(
_output_shapes
:?????????? 2
encoder/flatten_1/Reshape?
+encoder/latent_vector/MatMul/ReadVariableOpReadVariableOp4encoder_latent_vector_matmul_readvariableop_resource* 
_output_shapes
:
? ?*
dtype02-
+encoder/latent_vector/MatMul/ReadVariableOp?
encoder/latent_vector/MatMulMatMul"encoder/flatten_1/Reshape:output:03encoder/latent_vector/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
encoder/latent_vector/MatMul?
,encoder/latent_vector/BiasAdd/ReadVariableOpReadVariableOp5encoder_latent_vector_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02.
,encoder/latent_vector/BiasAdd/ReadVariableOp?
encoder/latent_vector/BiasAddBiasAdd&encoder/latent_vector/MatMul:product:04encoder/latent_vector/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
encoder/latent_vector/BiasAdd?
%decoder/dense_1/MatMul/ReadVariableOpReadVariableOp.decoder_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
?? *
dtype02'
%decoder/dense_1/MatMul/ReadVariableOp?
decoder/dense_1/MatMulMatMul&encoder/latent_vector/BiasAdd:output:0-decoder/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????? 2
decoder/dense_1/MatMul?
&decoder/dense_1/BiasAdd/ReadVariableOpReadVariableOp/decoder_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:? *
dtype02(
&decoder/dense_1/BiasAdd/ReadVariableOp?
decoder/dense_1/BiasAddBiasAdd decoder/dense_1/MatMul:product:0.decoder/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????? 2
decoder/dense_1/BiasAdd?
decoder/reshape_1/ShapeShape decoder/dense_1/BiasAdd:output:0*
T0*
_output_shapes
:2
decoder/reshape_1/Shape?
%decoder/reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%decoder/reshape_1/strided_slice/stack?
'decoder/reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'decoder/reshape_1/strided_slice/stack_1?
'decoder/reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'decoder/reshape_1/strided_slice/stack_2?
decoder/reshape_1/strided_sliceStridedSlice decoder/reshape_1/Shape:output:0.decoder/reshape_1/strided_slice/stack:output:00decoder/reshape_1/strided_slice/stack_1:output:00decoder/reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
decoder/reshape_1/strided_slice?
!decoder/reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2#
!decoder/reshape_1/Reshape/shape/1?
!decoder/reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2#
!decoder/reshape_1/Reshape/shape/2?
!decoder/reshape_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :?2#
!decoder/reshape_1/Reshape/shape/3?
decoder/reshape_1/Reshape/shapePack(decoder/reshape_1/strided_slice:output:0*decoder/reshape_1/Reshape/shape/1:output:0*decoder/reshape_1/Reshape/shape/2:output:0*decoder/reshape_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2!
decoder/reshape_1/Reshape/shape?
decoder/reshape_1/ReshapeReshape decoder/dense_1/BiasAdd:output:0(decoder/reshape_1/Reshape/shape:output:0*
T0*0
_output_shapes
:??????????2
decoder/reshape_1/Reshape?
 decoder/conv2d_transpose_3/ShapeShape"decoder/reshape_1/Reshape:output:0*
T0*
_output_shapes
:2"
 decoder/conv2d_transpose_3/Shape?
.decoder/conv2d_transpose_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 20
.decoder/conv2d_transpose_3/strided_slice/stack?
0decoder/conv2d_transpose_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0decoder/conv2d_transpose_3/strided_slice/stack_1?
0decoder/conv2d_transpose_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0decoder/conv2d_transpose_3/strided_slice/stack_2?
(decoder/conv2d_transpose_3/strided_sliceStridedSlice)decoder/conv2d_transpose_3/Shape:output:07decoder/conv2d_transpose_3/strided_slice/stack:output:09decoder/conv2d_transpose_3/strided_slice/stack_1:output:09decoder/conv2d_transpose_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(decoder/conv2d_transpose_3/strided_slice?
0decoder/conv2d_transpose_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:22
0decoder/conv2d_transpose_3/strided_slice_1/stack?
2decoder/conv2d_transpose_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2decoder/conv2d_transpose_3/strided_slice_1/stack_1?
2decoder/conv2d_transpose_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2decoder/conv2d_transpose_3/strided_slice_1/stack_2?
*decoder/conv2d_transpose_3/strided_slice_1StridedSlice)decoder/conv2d_transpose_3/Shape:output:09decoder/conv2d_transpose_3/strided_slice_1/stack:output:0;decoder/conv2d_transpose_3/strided_slice_1/stack_1:output:0;decoder/conv2d_transpose_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*decoder/conv2d_transpose_3/strided_slice_1?
0decoder/conv2d_transpose_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:22
0decoder/conv2d_transpose_3/strided_slice_2/stack?
2decoder/conv2d_transpose_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2decoder/conv2d_transpose_3/strided_slice_2/stack_1?
2decoder/conv2d_transpose_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2decoder/conv2d_transpose_3/strided_slice_2/stack_2?
*decoder/conv2d_transpose_3/strided_slice_2StridedSlice)decoder/conv2d_transpose_3/Shape:output:09decoder/conv2d_transpose_3/strided_slice_2/stack:output:0;decoder/conv2d_transpose_3/strided_slice_2/stack_1:output:0;decoder/conv2d_transpose_3/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*decoder/conv2d_transpose_3/strided_slice_2?
 decoder/conv2d_transpose_3/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2"
 decoder/conv2d_transpose_3/mul/y?
decoder/conv2d_transpose_3/mulMul3decoder/conv2d_transpose_3/strided_slice_1:output:0)decoder/conv2d_transpose_3/mul/y:output:0*
T0*
_output_shapes
: 2 
decoder/conv2d_transpose_3/mul?
"decoder/conv2d_transpose_3/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2$
"decoder/conv2d_transpose_3/mul_1/y?
 decoder/conv2d_transpose_3/mul_1Mul3decoder/conv2d_transpose_3/strided_slice_2:output:0+decoder/conv2d_transpose_3/mul_1/y:output:0*
T0*
_output_shapes
: 2"
 decoder/conv2d_transpose_3/mul_1?
"decoder/conv2d_transpose_3/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2$
"decoder/conv2d_transpose_3/stack/3?
 decoder/conv2d_transpose_3/stackPack1decoder/conv2d_transpose_3/strided_slice:output:0"decoder/conv2d_transpose_3/mul:z:0$decoder/conv2d_transpose_3/mul_1:z:0+decoder/conv2d_transpose_3/stack/3:output:0*
N*
T0*
_output_shapes
:2"
 decoder/conv2d_transpose_3/stack?
0decoder/conv2d_transpose_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0decoder/conv2d_transpose_3/strided_slice_3/stack?
2decoder/conv2d_transpose_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2decoder/conv2d_transpose_3/strided_slice_3/stack_1?
2decoder/conv2d_transpose_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2decoder/conv2d_transpose_3/strided_slice_3/stack_2?
*decoder/conv2d_transpose_3/strided_slice_3StridedSlice)decoder/conv2d_transpose_3/stack:output:09decoder/conv2d_transpose_3/strided_slice_3/stack:output:0;decoder/conv2d_transpose_3/strided_slice_3/stack_1:output:0;decoder/conv2d_transpose_3/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*decoder/conv2d_transpose_3/strided_slice_3?
:decoder/conv2d_transpose_3/conv2d_transpose/ReadVariableOpReadVariableOpCdecoder_conv2d_transpose_3_conv2d_transpose_readvariableop_resource*'
_output_shapes
:?*
dtype02<
:decoder/conv2d_transpose_3/conv2d_transpose/ReadVariableOp?
+decoder/conv2d_transpose_3/conv2d_transposeConv2DBackpropInput)decoder/conv2d_transpose_3/stack:output:0Bdecoder/conv2d_transpose_3/conv2d_transpose/ReadVariableOp:value:0"decoder/reshape_1/Reshape:output:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2-
+decoder/conv2d_transpose_3/conv2d_transpose?
1decoder/conv2d_transpose_3/BiasAdd/ReadVariableOpReadVariableOp:decoder_conv2d_transpose_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype023
1decoder/conv2d_transpose_3/BiasAdd/ReadVariableOp?
"decoder/conv2d_transpose_3/BiasAddBiasAdd4decoder/conv2d_transpose_3/conv2d_transpose:output:09decoder/conv2d_transpose_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2$
"decoder/conv2d_transpose_3/BiasAdd?
decoder/conv2d_transpose_3/ReluRelu+decoder/conv2d_transpose_3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2!
decoder/conv2d_transpose_3/Relu?
 decoder/conv2d_transpose_4/ShapeShape-decoder/conv2d_transpose_3/Relu:activations:0*
T0*
_output_shapes
:2"
 decoder/conv2d_transpose_4/Shape?
.decoder/conv2d_transpose_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 20
.decoder/conv2d_transpose_4/strided_slice/stack?
0decoder/conv2d_transpose_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0decoder/conv2d_transpose_4/strided_slice/stack_1?
0decoder/conv2d_transpose_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0decoder/conv2d_transpose_4/strided_slice/stack_2?
(decoder/conv2d_transpose_4/strided_sliceStridedSlice)decoder/conv2d_transpose_4/Shape:output:07decoder/conv2d_transpose_4/strided_slice/stack:output:09decoder/conv2d_transpose_4/strided_slice/stack_1:output:09decoder/conv2d_transpose_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(decoder/conv2d_transpose_4/strided_slice?
0decoder/conv2d_transpose_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:22
0decoder/conv2d_transpose_4/strided_slice_1/stack?
2decoder/conv2d_transpose_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2decoder/conv2d_transpose_4/strided_slice_1/stack_1?
2decoder/conv2d_transpose_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2decoder/conv2d_transpose_4/strided_slice_1/stack_2?
*decoder/conv2d_transpose_4/strided_slice_1StridedSlice)decoder/conv2d_transpose_4/Shape:output:09decoder/conv2d_transpose_4/strided_slice_1/stack:output:0;decoder/conv2d_transpose_4/strided_slice_1/stack_1:output:0;decoder/conv2d_transpose_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*decoder/conv2d_transpose_4/strided_slice_1?
0decoder/conv2d_transpose_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:22
0decoder/conv2d_transpose_4/strided_slice_2/stack?
2decoder/conv2d_transpose_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2decoder/conv2d_transpose_4/strided_slice_2/stack_1?
2decoder/conv2d_transpose_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2decoder/conv2d_transpose_4/strided_slice_2/stack_2?
*decoder/conv2d_transpose_4/strided_slice_2StridedSlice)decoder/conv2d_transpose_4/Shape:output:09decoder/conv2d_transpose_4/strided_slice_2/stack:output:0;decoder/conv2d_transpose_4/strided_slice_2/stack_1:output:0;decoder/conv2d_transpose_4/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*decoder/conv2d_transpose_4/strided_slice_2?
 decoder/conv2d_transpose_4/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2"
 decoder/conv2d_transpose_4/mul/y?
decoder/conv2d_transpose_4/mulMul3decoder/conv2d_transpose_4/strided_slice_1:output:0)decoder/conv2d_transpose_4/mul/y:output:0*
T0*
_output_shapes
: 2 
decoder/conv2d_transpose_4/mul?
"decoder/conv2d_transpose_4/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2$
"decoder/conv2d_transpose_4/mul_1/y?
 decoder/conv2d_transpose_4/mul_1Mul3decoder/conv2d_transpose_4/strided_slice_2:output:0+decoder/conv2d_transpose_4/mul_1/y:output:0*
T0*
_output_shapes
: 2"
 decoder/conv2d_transpose_4/mul_1?
"decoder/conv2d_transpose_4/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2$
"decoder/conv2d_transpose_4/stack/3?
 decoder/conv2d_transpose_4/stackPack1decoder/conv2d_transpose_4/strided_slice:output:0"decoder/conv2d_transpose_4/mul:z:0$decoder/conv2d_transpose_4/mul_1:z:0+decoder/conv2d_transpose_4/stack/3:output:0*
N*
T0*
_output_shapes
:2"
 decoder/conv2d_transpose_4/stack?
0decoder/conv2d_transpose_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0decoder/conv2d_transpose_4/strided_slice_3/stack?
2decoder/conv2d_transpose_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2decoder/conv2d_transpose_4/strided_slice_3/stack_1?
2decoder/conv2d_transpose_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2decoder/conv2d_transpose_4/strided_slice_3/stack_2?
*decoder/conv2d_transpose_4/strided_slice_3StridedSlice)decoder/conv2d_transpose_4/stack:output:09decoder/conv2d_transpose_4/strided_slice_3/stack:output:0;decoder/conv2d_transpose_4/strided_slice_3/stack_1:output:0;decoder/conv2d_transpose_4/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*decoder/conv2d_transpose_4/strided_slice_3?
:decoder/conv2d_transpose_4/conv2d_transpose/ReadVariableOpReadVariableOpCdecoder_conv2d_transpose_4_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02<
:decoder/conv2d_transpose_4/conv2d_transpose/ReadVariableOp?
+decoder/conv2d_transpose_4/conv2d_transposeConv2DBackpropInput)decoder/conv2d_transpose_4/stack:output:0Bdecoder/conv2d_transpose_4/conv2d_transpose/ReadVariableOp:value:0-decoder/conv2d_transpose_3/Relu:activations:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2-
+decoder/conv2d_transpose_4/conv2d_transpose?
1decoder/conv2d_transpose_4/BiasAdd/ReadVariableOpReadVariableOp:decoder_conv2d_transpose_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype023
1decoder/conv2d_transpose_4/BiasAdd/ReadVariableOp?
"decoder/conv2d_transpose_4/BiasAddBiasAdd4decoder/conv2d_transpose_4/conv2d_transpose:output:09decoder/conv2d_transpose_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2$
"decoder/conv2d_transpose_4/BiasAdd?
decoder/conv2d_transpose_4/ReluRelu+decoder/conv2d_transpose_4/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2!
decoder/conv2d_transpose_4/Relu?
 decoder/conv2d_transpose_5/ShapeShape-decoder/conv2d_transpose_4/Relu:activations:0*
T0*
_output_shapes
:2"
 decoder/conv2d_transpose_5/Shape?
.decoder/conv2d_transpose_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 20
.decoder/conv2d_transpose_5/strided_slice/stack?
0decoder/conv2d_transpose_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0decoder/conv2d_transpose_5/strided_slice/stack_1?
0decoder/conv2d_transpose_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0decoder/conv2d_transpose_5/strided_slice/stack_2?
(decoder/conv2d_transpose_5/strided_sliceStridedSlice)decoder/conv2d_transpose_5/Shape:output:07decoder/conv2d_transpose_5/strided_slice/stack:output:09decoder/conv2d_transpose_5/strided_slice/stack_1:output:09decoder/conv2d_transpose_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(decoder/conv2d_transpose_5/strided_slice?
0decoder/conv2d_transpose_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:22
0decoder/conv2d_transpose_5/strided_slice_1/stack?
2decoder/conv2d_transpose_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2decoder/conv2d_transpose_5/strided_slice_1/stack_1?
2decoder/conv2d_transpose_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2decoder/conv2d_transpose_5/strided_slice_1/stack_2?
*decoder/conv2d_transpose_5/strided_slice_1StridedSlice)decoder/conv2d_transpose_5/Shape:output:09decoder/conv2d_transpose_5/strided_slice_1/stack:output:0;decoder/conv2d_transpose_5/strided_slice_1/stack_1:output:0;decoder/conv2d_transpose_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*decoder/conv2d_transpose_5/strided_slice_1?
0decoder/conv2d_transpose_5/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:22
0decoder/conv2d_transpose_5/strided_slice_2/stack?
2decoder/conv2d_transpose_5/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2decoder/conv2d_transpose_5/strided_slice_2/stack_1?
2decoder/conv2d_transpose_5/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2decoder/conv2d_transpose_5/strided_slice_2/stack_2?
*decoder/conv2d_transpose_5/strided_slice_2StridedSlice)decoder/conv2d_transpose_5/Shape:output:09decoder/conv2d_transpose_5/strided_slice_2/stack:output:0;decoder/conv2d_transpose_5/strided_slice_2/stack_1:output:0;decoder/conv2d_transpose_5/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*decoder/conv2d_transpose_5/strided_slice_2?
 decoder/conv2d_transpose_5/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2"
 decoder/conv2d_transpose_5/mul/y?
decoder/conv2d_transpose_5/mulMul3decoder/conv2d_transpose_5/strided_slice_1:output:0)decoder/conv2d_transpose_5/mul/y:output:0*
T0*
_output_shapes
: 2 
decoder/conv2d_transpose_5/mul?
"decoder/conv2d_transpose_5/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2$
"decoder/conv2d_transpose_5/mul_1/y?
 decoder/conv2d_transpose_5/mul_1Mul3decoder/conv2d_transpose_5/strided_slice_2:output:0+decoder/conv2d_transpose_5/mul_1/y:output:0*
T0*
_output_shapes
: 2"
 decoder/conv2d_transpose_5/mul_1?
"decoder/conv2d_transpose_5/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2$
"decoder/conv2d_transpose_5/stack/3?
 decoder/conv2d_transpose_5/stackPack1decoder/conv2d_transpose_5/strided_slice:output:0"decoder/conv2d_transpose_5/mul:z:0$decoder/conv2d_transpose_5/mul_1:z:0+decoder/conv2d_transpose_5/stack/3:output:0*
N*
T0*
_output_shapes
:2"
 decoder/conv2d_transpose_5/stack?
0decoder/conv2d_transpose_5/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0decoder/conv2d_transpose_5/strided_slice_3/stack?
2decoder/conv2d_transpose_5/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2decoder/conv2d_transpose_5/strided_slice_3/stack_1?
2decoder/conv2d_transpose_5/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2decoder/conv2d_transpose_5/strided_slice_3/stack_2?
*decoder/conv2d_transpose_5/strided_slice_3StridedSlice)decoder/conv2d_transpose_5/stack:output:09decoder/conv2d_transpose_5/strided_slice_3/stack:output:0;decoder/conv2d_transpose_5/strided_slice_3/stack_1:output:0;decoder/conv2d_transpose_5/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*decoder/conv2d_transpose_5/strided_slice_3?
:decoder/conv2d_transpose_5/conv2d_transpose/ReadVariableOpReadVariableOpCdecoder_conv2d_transpose_5_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02<
:decoder/conv2d_transpose_5/conv2d_transpose/ReadVariableOp?
+decoder/conv2d_transpose_5/conv2d_transposeConv2DBackpropInput)decoder/conv2d_transpose_5/stack:output:0Bdecoder/conv2d_transpose_5/conv2d_transpose/ReadVariableOp:value:0-decoder/conv2d_transpose_4/Relu:activations:0*
T0*/
_output_shapes
:?????????  *
paddingSAME*
strides
2-
+decoder/conv2d_transpose_5/conv2d_transpose?
1decoder/conv2d_transpose_5/BiasAdd/ReadVariableOpReadVariableOp:decoder_conv2d_transpose_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype023
1decoder/conv2d_transpose_5/BiasAdd/ReadVariableOp?
"decoder/conv2d_transpose_5/BiasAddBiasAdd4decoder/conv2d_transpose_5/conv2d_transpose:output:09decoder/conv2d_transpose_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  2$
"decoder/conv2d_transpose_5/BiasAdd?
decoder/conv2d_transpose_5/ReluRelu+decoder/conv2d_transpose_5/BiasAdd:output:0*
T0*/
_output_shapes
:?????????  2!
decoder/conv2d_transpose_5/Relu?
decoder/decoder_output/ShapeShape-decoder/conv2d_transpose_5/Relu:activations:0*
T0*
_output_shapes
:2
decoder/decoder_output/Shape?
*decoder/decoder_output/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*decoder/decoder_output/strided_slice/stack?
,decoder/decoder_output/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,decoder/decoder_output/strided_slice/stack_1?
,decoder/decoder_output/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,decoder/decoder_output/strided_slice/stack_2?
$decoder/decoder_output/strided_sliceStridedSlice%decoder/decoder_output/Shape:output:03decoder/decoder_output/strided_slice/stack:output:05decoder/decoder_output/strided_slice/stack_1:output:05decoder/decoder_output/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$decoder/decoder_output/strided_slice?
,decoder/decoder_output/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2.
,decoder/decoder_output/strided_slice_1/stack?
.decoder/decoder_output/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.decoder/decoder_output/strided_slice_1/stack_1?
.decoder/decoder_output/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.decoder/decoder_output/strided_slice_1/stack_2?
&decoder/decoder_output/strided_slice_1StridedSlice%decoder/decoder_output/Shape:output:05decoder/decoder_output/strided_slice_1/stack:output:07decoder/decoder_output/strided_slice_1/stack_1:output:07decoder/decoder_output/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&decoder/decoder_output/strided_slice_1?
,decoder/decoder_output/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2.
,decoder/decoder_output/strided_slice_2/stack?
.decoder/decoder_output/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.decoder/decoder_output/strided_slice_2/stack_1?
.decoder/decoder_output/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.decoder/decoder_output/strided_slice_2/stack_2?
&decoder/decoder_output/strided_slice_2StridedSlice%decoder/decoder_output/Shape:output:05decoder/decoder_output/strided_slice_2/stack:output:07decoder/decoder_output/strided_slice_2/stack_1:output:07decoder/decoder_output/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&decoder/decoder_output/strided_slice_2~
decoder/decoder_output/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
decoder/decoder_output/mul/y?
decoder/decoder_output/mulMul/decoder/decoder_output/strided_slice_1:output:0%decoder/decoder_output/mul/y:output:0*
T0*
_output_shapes
: 2
decoder/decoder_output/mul?
decoder/decoder_output/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2 
decoder/decoder_output/mul_1/y?
decoder/decoder_output/mul_1Mul/decoder/decoder_output/strided_slice_2:output:0'decoder/decoder_output/mul_1/y:output:0*
T0*
_output_shapes
: 2
decoder/decoder_output/mul_1?
decoder/decoder_output/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2 
decoder/decoder_output/stack/3?
decoder/decoder_output/stackPack-decoder/decoder_output/strided_slice:output:0decoder/decoder_output/mul:z:0 decoder/decoder_output/mul_1:z:0'decoder/decoder_output/stack/3:output:0*
N*
T0*
_output_shapes
:2
decoder/decoder_output/stack?
,decoder/decoder_output/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,decoder/decoder_output/strided_slice_3/stack?
.decoder/decoder_output/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.decoder/decoder_output/strided_slice_3/stack_1?
.decoder/decoder_output/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.decoder/decoder_output/strided_slice_3/stack_2?
&decoder/decoder_output/strided_slice_3StridedSlice%decoder/decoder_output/stack:output:05decoder/decoder_output/strided_slice_3/stack:output:07decoder/decoder_output/strided_slice_3/stack_1:output:07decoder/decoder_output/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&decoder/decoder_output/strided_slice_3?
6decoder/decoder_output/conv2d_transpose/ReadVariableOpReadVariableOp?decoder_decoder_output_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype028
6decoder/decoder_output/conv2d_transpose/ReadVariableOp?
'decoder/decoder_output/conv2d_transposeConv2DBackpropInput%decoder/decoder_output/stack:output:0>decoder/decoder_output/conv2d_transpose/ReadVariableOp:value:0-decoder/conv2d_transpose_5/Relu:activations:0*
T0*/
_output_shapes
:?????????  *
paddingSAME*
strides
2)
'decoder/decoder_output/conv2d_transpose?
-decoder/decoder_output/BiasAdd/ReadVariableOpReadVariableOp6decoder_decoder_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-decoder/decoder_output/BiasAdd/ReadVariableOp?
decoder/decoder_output/BiasAddBiasAdd0decoder/decoder_output/conv2d_transpose:output:05decoder/decoder_output/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  2 
decoder/decoder_output/BiasAdd?
decoder/decoder_output/SigmoidSigmoid'decoder/decoder_output/BiasAdd:output:0*
T0*/
_output_shapes
:?????????  2 
decoder/decoder_output/Sigmoid?
IdentityIdentity"decoder/decoder_output/Sigmoid:y:02^decoder/conv2d_transpose_3/BiasAdd/ReadVariableOp;^decoder/conv2d_transpose_3/conv2d_transpose/ReadVariableOp2^decoder/conv2d_transpose_4/BiasAdd/ReadVariableOp;^decoder/conv2d_transpose_4/conv2d_transpose/ReadVariableOp2^decoder/conv2d_transpose_5/BiasAdd/ReadVariableOp;^decoder/conv2d_transpose_5/conv2d_transpose/ReadVariableOp.^decoder/decoder_output/BiasAdd/ReadVariableOp7^decoder/decoder_output/conv2d_transpose/ReadVariableOp'^decoder/dense_1/BiasAdd/ReadVariableOp&^decoder/dense_1/MatMul/ReadVariableOp(^encoder/conv2d_3/BiasAdd/ReadVariableOp'^encoder/conv2d_3/Conv2D/ReadVariableOp(^encoder/conv2d_4/BiasAdd/ReadVariableOp'^encoder/conv2d_4/Conv2D/ReadVariableOp(^encoder/conv2d_5/BiasAdd/ReadVariableOp'^encoder/conv2d_5/Conv2D/ReadVariableOp-^encoder/latent_vector/BiasAdd/ReadVariableOp,^encoder/latent_vector/MatMul/ReadVariableOp*
T0*/
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*v
_input_shapese
c:?????????  ::::::::::::::::::2f
1decoder/conv2d_transpose_3/BiasAdd/ReadVariableOp1decoder/conv2d_transpose_3/BiasAdd/ReadVariableOp2x
:decoder/conv2d_transpose_3/conv2d_transpose/ReadVariableOp:decoder/conv2d_transpose_3/conv2d_transpose/ReadVariableOp2f
1decoder/conv2d_transpose_4/BiasAdd/ReadVariableOp1decoder/conv2d_transpose_4/BiasAdd/ReadVariableOp2x
:decoder/conv2d_transpose_4/conv2d_transpose/ReadVariableOp:decoder/conv2d_transpose_4/conv2d_transpose/ReadVariableOp2f
1decoder/conv2d_transpose_5/BiasAdd/ReadVariableOp1decoder/conv2d_transpose_5/BiasAdd/ReadVariableOp2x
:decoder/conv2d_transpose_5/conv2d_transpose/ReadVariableOp:decoder/conv2d_transpose_5/conv2d_transpose/ReadVariableOp2^
-decoder/decoder_output/BiasAdd/ReadVariableOp-decoder/decoder_output/BiasAdd/ReadVariableOp2p
6decoder/decoder_output/conv2d_transpose/ReadVariableOp6decoder/decoder_output/conv2d_transpose/ReadVariableOp2P
&decoder/dense_1/BiasAdd/ReadVariableOp&decoder/dense_1/BiasAdd/ReadVariableOp2N
%decoder/dense_1/MatMul/ReadVariableOp%decoder/dense_1/MatMul/ReadVariableOp2R
'encoder/conv2d_3/BiasAdd/ReadVariableOp'encoder/conv2d_3/BiasAdd/ReadVariableOp2P
&encoder/conv2d_3/Conv2D/ReadVariableOp&encoder/conv2d_3/Conv2D/ReadVariableOp2R
'encoder/conv2d_4/BiasAdd/ReadVariableOp'encoder/conv2d_4/BiasAdd/ReadVariableOp2P
&encoder/conv2d_4/Conv2D/ReadVariableOp&encoder/conv2d_4/Conv2D/ReadVariableOp2R
'encoder/conv2d_5/BiasAdd/ReadVariableOp'encoder/conv2d_5/BiasAdd/ReadVariableOp2P
&encoder/conv2d_5/Conv2D/ReadVariableOp&encoder/conv2d_5/Conv2D/ReadVariableOp2\
,encoder/latent_vector/BiasAdd/ReadVariableOp,encoder/latent_vector/BiasAdd/ReadVariableOp2Z
+encoder/latent_vector/MatMul/ReadVariableOp+encoder/latent_vector/MatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
?

?
'__inference_encoder_layer_call_fn_88917

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8*
Tin
2	*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????*-
config_proto

GPU

CPU2*0J 8*K
fFRD
B__inference_encoder_layer_call_and_return_conditional_losses_878572
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????  ::::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?
?
'__inference_decoder_layer_call_fn_88187
decoder_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldecoder_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+???????????????????????????*-
config_proto

GPU

CPU2*0J 8*K
fFRD
B__inference_decoder_layer_call_and_return_conditional_losses_881742
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*O
_input_shapes>
<:??????????::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:- )
'
_user_specified_namedecoder_input
??
?
B__inference_decoder_layer_call_and_return_conditional_losses_89070

inputs*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource?
;conv2d_transpose_3_conv2d_transpose_readvariableop_resource6
2conv2d_transpose_3_biasadd_readvariableop_resource?
;conv2d_transpose_4_conv2d_transpose_readvariableop_resource6
2conv2d_transpose_4_biasadd_readvariableop_resource?
;conv2d_transpose_5_conv2d_transpose_readvariableop_resource6
2conv2d_transpose_5_biasadd_readvariableop_resource;
7decoder_output_conv2d_transpose_readvariableop_resource2
.decoder_output_biasadd_readvariableop_resource
identity??)conv2d_transpose_3/BiasAdd/ReadVariableOp?2conv2d_transpose_3/conv2d_transpose/ReadVariableOp?)conv2d_transpose_4/BiasAdd/ReadVariableOp?2conv2d_transpose_4/conv2d_transpose/ReadVariableOp?)conv2d_transpose_5/BiasAdd/ReadVariableOp?2conv2d_transpose_5/conv2d_transpose/ReadVariableOp?%decoder_output/BiasAdd/ReadVariableOp?.decoder_output/conv2d_transpose/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
?? *
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMulinputs%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????? 2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:? *
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????? 2
dense_1/BiasAddj
reshape_1/ShapeShapedense_1/BiasAdd:output:0*
T0*
_output_shapes
:2
reshape_1/Shape?
reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_1/strided_slice/stack?
reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_1/strided_slice/stack_1?
reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_1/strided_slice/stack_2?
reshape_1/strided_sliceStridedSlicereshape_1/Shape:output:0&reshape_1/strided_slice/stack:output:0(reshape_1/strided_slice/stack_1:output:0(reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_1/strided_slicex
reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_1/Reshape/shape/1x
reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_1/Reshape/shape/2y
reshape_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :?2
reshape_1/Reshape/shape/3?
reshape_1/Reshape/shapePack reshape_1/strided_slice:output:0"reshape_1/Reshape/shape/1:output:0"reshape_1/Reshape/shape/2:output:0"reshape_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape_1/Reshape/shape?
reshape_1/ReshapeReshapedense_1/BiasAdd:output:0 reshape_1/Reshape/shape:output:0*
T0*0
_output_shapes
:??????????2
reshape_1/Reshape~
conv2d_transpose_3/ShapeShapereshape_1/Reshape:output:0*
T0*
_output_shapes
:2
conv2d_transpose_3/Shape?
&conv2d_transpose_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose_3/strided_slice/stack?
(conv2d_transpose_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_3/strided_slice/stack_1?
(conv2d_transpose_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_3/strided_slice/stack_2?
 conv2d_transpose_3/strided_sliceStridedSlice!conv2d_transpose_3/Shape:output:0/conv2d_transpose_3/strided_slice/stack:output:01conv2d_transpose_3/strided_slice/stack_1:output:01conv2d_transpose_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose_3/strided_slice?
(conv2d_transpose_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_3/strided_slice_1/stack?
*conv2d_transpose_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_3/strided_slice_1/stack_1?
*conv2d_transpose_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_3/strided_slice_1/stack_2?
"conv2d_transpose_3/strided_slice_1StridedSlice!conv2d_transpose_3/Shape:output:01conv2d_transpose_3/strided_slice_1/stack:output:03conv2d_transpose_3/strided_slice_1/stack_1:output:03conv2d_transpose_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_3/strided_slice_1?
(conv2d_transpose_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_3/strided_slice_2/stack?
*conv2d_transpose_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_3/strided_slice_2/stack_1?
*conv2d_transpose_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_3/strided_slice_2/stack_2?
"conv2d_transpose_3/strided_slice_2StridedSlice!conv2d_transpose_3/Shape:output:01conv2d_transpose_3/strided_slice_2/stack:output:03conv2d_transpose_3/strided_slice_2/stack_1:output:03conv2d_transpose_3/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_3/strided_slice_2v
conv2d_transpose_3/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_3/mul/y?
conv2d_transpose_3/mulMul+conv2d_transpose_3/strided_slice_1:output:0!conv2d_transpose_3/mul/y:output:0*
T0*
_output_shapes
: 2
conv2d_transpose_3/mulz
conv2d_transpose_3/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_3/mul_1/y?
conv2d_transpose_3/mul_1Mul+conv2d_transpose_3/strided_slice_2:output:0#conv2d_transpose_3/mul_1/y:output:0*
T0*
_output_shapes
: 2
conv2d_transpose_3/mul_1z
conv2d_transpose_3/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_3/stack/3?
conv2d_transpose_3/stackPack)conv2d_transpose_3/strided_slice:output:0conv2d_transpose_3/mul:z:0conv2d_transpose_3/mul_1:z:0#conv2d_transpose_3/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_3/stack?
(conv2d_transpose_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(conv2d_transpose_3/strided_slice_3/stack?
*conv2d_transpose_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_3/strided_slice_3/stack_1?
*conv2d_transpose_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_3/strided_slice_3/stack_2?
"conv2d_transpose_3/strided_slice_3StridedSlice!conv2d_transpose_3/stack:output:01conv2d_transpose_3/strided_slice_3/stack:output:03conv2d_transpose_3/strided_slice_3/stack_1:output:03conv2d_transpose_3/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_3/strided_slice_3?
2conv2d_transpose_3/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_3_conv2d_transpose_readvariableop_resource*'
_output_shapes
:?*
dtype024
2conv2d_transpose_3/conv2d_transpose/ReadVariableOp?
#conv2d_transpose_3/conv2d_transposeConv2DBackpropInput!conv2d_transpose_3/stack:output:0:conv2d_transpose_3/conv2d_transpose/ReadVariableOp:value:0reshape_1/Reshape:output:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2%
#conv2d_transpose_3/conv2d_transpose?
)conv2d_transpose_3/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)conv2d_transpose_3/BiasAdd/ReadVariableOp?
conv2d_transpose_3/BiasAddBiasAdd,conv2d_transpose_3/conv2d_transpose:output:01conv2d_transpose_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d_transpose_3/BiasAdd?
conv2d_transpose_3/ReluRelu#conv2d_transpose_3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
conv2d_transpose_3/Relu?
conv2d_transpose_4/ShapeShape%conv2d_transpose_3/Relu:activations:0*
T0*
_output_shapes
:2
conv2d_transpose_4/Shape?
&conv2d_transpose_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose_4/strided_slice/stack?
(conv2d_transpose_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_4/strided_slice/stack_1?
(conv2d_transpose_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_4/strided_slice/stack_2?
 conv2d_transpose_4/strided_sliceStridedSlice!conv2d_transpose_4/Shape:output:0/conv2d_transpose_4/strided_slice/stack:output:01conv2d_transpose_4/strided_slice/stack_1:output:01conv2d_transpose_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose_4/strided_slice?
(conv2d_transpose_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_4/strided_slice_1/stack?
*conv2d_transpose_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_4/strided_slice_1/stack_1?
*conv2d_transpose_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_4/strided_slice_1/stack_2?
"conv2d_transpose_4/strided_slice_1StridedSlice!conv2d_transpose_4/Shape:output:01conv2d_transpose_4/strided_slice_1/stack:output:03conv2d_transpose_4/strided_slice_1/stack_1:output:03conv2d_transpose_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_4/strided_slice_1?
(conv2d_transpose_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_4/strided_slice_2/stack?
*conv2d_transpose_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_4/strided_slice_2/stack_1?
*conv2d_transpose_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_4/strided_slice_2/stack_2?
"conv2d_transpose_4/strided_slice_2StridedSlice!conv2d_transpose_4/Shape:output:01conv2d_transpose_4/strided_slice_2/stack:output:03conv2d_transpose_4/strided_slice_2/stack_1:output:03conv2d_transpose_4/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_4/strided_slice_2v
conv2d_transpose_4/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_4/mul/y?
conv2d_transpose_4/mulMul+conv2d_transpose_4/strided_slice_1:output:0!conv2d_transpose_4/mul/y:output:0*
T0*
_output_shapes
: 2
conv2d_transpose_4/mulz
conv2d_transpose_4/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_4/mul_1/y?
conv2d_transpose_4/mul_1Mul+conv2d_transpose_4/strided_slice_2:output:0#conv2d_transpose_4/mul_1/y:output:0*
T0*
_output_shapes
: 2
conv2d_transpose_4/mul_1z
conv2d_transpose_4/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_4/stack/3?
conv2d_transpose_4/stackPack)conv2d_transpose_4/strided_slice:output:0conv2d_transpose_4/mul:z:0conv2d_transpose_4/mul_1:z:0#conv2d_transpose_4/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_4/stack?
(conv2d_transpose_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(conv2d_transpose_4/strided_slice_3/stack?
*conv2d_transpose_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_4/strided_slice_3/stack_1?
*conv2d_transpose_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_4/strided_slice_3/stack_2?
"conv2d_transpose_4/strided_slice_3StridedSlice!conv2d_transpose_4/stack:output:01conv2d_transpose_4/strided_slice_3/stack:output:03conv2d_transpose_4/strided_slice_3/stack_1:output:03conv2d_transpose_4/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_4/strided_slice_3?
2conv2d_transpose_4/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_4_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype024
2conv2d_transpose_4/conv2d_transpose/ReadVariableOp?
#conv2d_transpose_4/conv2d_transposeConv2DBackpropInput!conv2d_transpose_4/stack:output:0:conv2d_transpose_4/conv2d_transpose/ReadVariableOp:value:0%conv2d_transpose_3/Relu:activations:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2%
#conv2d_transpose_4/conv2d_transpose?
)conv2d_transpose_4/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)conv2d_transpose_4/BiasAdd/ReadVariableOp?
conv2d_transpose_4/BiasAddBiasAdd,conv2d_transpose_4/conv2d_transpose:output:01conv2d_transpose_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d_transpose_4/BiasAdd?
conv2d_transpose_4/ReluRelu#conv2d_transpose_4/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
conv2d_transpose_4/Relu?
conv2d_transpose_5/ShapeShape%conv2d_transpose_4/Relu:activations:0*
T0*
_output_shapes
:2
conv2d_transpose_5/Shape?
&conv2d_transpose_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose_5/strided_slice/stack?
(conv2d_transpose_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_5/strided_slice/stack_1?
(conv2d_transpose_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_5/strided_slice/stack_2?
 conv2d_transpose_5/strided_sliceStridedSlice!conv2d_transpose_5/Shape:output:0/conv2d_transpose_5/strided_slice/stack:output:01conv2d_transpose_5/strided_slice/stack_1:output:01conv2d_transpose_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose_5/strided_slice?
(conv2d_transpose_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_5/strided_slice_1/stack?
*conv2d_transpose_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_5/strided_slice_1/stack_1?
*conv2d_transpose_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_5/strided_slice_1/stack_2?
"conv2d_transpose_5/strided_slice_1StridedSlice!conv2d_transpose_5/Shape:output:01conv2d_transpose_5/strided_slice_1/stack:output:03conv2d_transpose_5/strided_slice_1/stack_1:output:03conv2d_transpose_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_5/strided_slice_1?
(conv2d_transpose_5/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_5/strided_slice_2/stack?
*conv2d_transpose_5/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_5/strided_slice_2/stack_1?
*conv2d_transpose_5/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_5/strided_slice_2/stack_2?
"conv2d_transpose_5/strided_slice_2StridedSlice!conv2d_transpose_5/Shape:output:01conv2d_transpose_5/strided_slice_2/stack:output:03conv2d_transpose_5/strided_slice_2/stack_1:output:03conv2d_transpose_5/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_5/strided_slice_2v
conv2d_transpose_5/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_5/mul/y?
conv2d_transpose_5/mulMul+conv2d_transpose_5/strided_slice_1:output:0!conv2d_transpose_5/mul/y:output:0*
T0*
_output_shapes
: 2
conv2d_transpose_5/mulz
conv2d_transpose_5/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_5/mul_1/y?
conv2d_transpose_5/mul_1Mul+conv2d_transpose_5/strided_slice_2:output:0#conv2d_transpose_5/mul_1/y:output:0*
T0*
_output_shapes
: 2
conv2d_transpose_5/mul_1z
conv2d_transpose_5/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_5/stack/3?
conv2d_transpose_5/stackPack)conv2d_transpose_5/strided_slice:output:0conv2d_transpose_5/mul:z:0conv2d_transpose_5/mul_1:z:0#conv2d_transpose_5/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_5/stack?
(conv2d_transpose_5/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(conv2d_transpose_5/strided_slice_3/stack?
*conv2d_transpose_5/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_5/strided_slice_3/stack_1?
*conv2d_transpose_5/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_5/strided_slice_3/stack_2?
"conv2d_transpose_5/strided_slice_3StridedSlice!conv2d_transpose_5/stack:output:01conv2d_transpose_5/strided_slice_3/stack:output:03conv2d_transpose_5/strided_slice_3/stack_1:output:03conv2d_transpose_5/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_5/strided_slice_3?
2conv2d_transpose_5/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_5_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype024
2conv2d_transpose_5/conv2d_transpose/ReadVariableOp?
#conv2d_transpose_5/conv2d_transposeConv2DBackpropInput!conv2d_transpose_5/stack:output:0:conv2d_transpose_5/conv2d_transpose/ReadVariableOp:value:0%conv2d_transpose_4/Relu:activations:0*
T0*/
_output_shapes
:?????????  *
paddingSAME*
strides
2%
#conv2d_transpose_5/conv2d_transpose?
)conv2d_transpose_5/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)conv2d_transpose_5/BiasAdd/ReadVariableOp?
conv2d_transpose_5/BiasAddBiasAdd,conv2d_transpose_5/conv2d_transpose:output:01conv2d_transpose_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  2
conv2d_transpose_5/BiasAdd?
conv2d_transpose_5/ReluRelu#conv2d_transpose_5/BiasAdd:output:0*
T0*/
_output_shapes
:?????????  2
conv2d_transpose_5/Relu?
decoder_output/ShapeShape%conv2d_transpose_5/Relu:activations:0*
T0*
_output_shapes
:2
decoder_output/Shape?
"decoder_output/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"decoder_output/strided_slice/stack?
$decoder_output/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$decoder_output/strided_slice/stack_1?
$decoder_output/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$decoder_output/strided_slice/stack_2?
decoder_output/strided_sliceStridedSlicedecoder_output/Shape:output:0+decoder_output/strided_slice/stack:output:0-decoder_output/strided_slice/stack_1:output:0-decoder_output/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
decoder_output/strided_slice?
$decoder_output/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2&
$decoder_output/strided_slice_1/stack?
&decoder_output/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&decoder_output/strided_slice_1/stack_1?
&decoder_output/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&decoder_output/strided_slice_1/stack_2?
decoder_output/strided_slice_1StridedSlicedecoder_output/Shape:output:0-decoder_output/strided_slice_1/stack:output:0/decoder_output/strided_slice_1/stack_1:output:0/decoder_output/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
decoder_output/strided_slice_1?
$decoder_output/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2&
$decoder_output/strided_slice_2/stack?
&decoder_output/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&decoder_output/strided_slice_2/stack_1?
&decoder_output/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&decoder_output/strided_slice_2/stack_2?
decoder_output/strided_slice_2StridedSlicedecoder_output/Shape:output:0-decoder_output/strided_slice_2/stack:output:0/decoder_output/strided_slice_2/stack_1:output:0/decoder_output/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
decoder_output/strided_slice_2n
decoder_output/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
decoder_output/mul/y?
decoder_output/mulMul'decoder_output/strided_slice_1:output:0decoder_output/mul/y:output:0*
T0*
_output_shapes
: 2
decoder_output/mulr
decoder_output/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
decoder_output/mul_1/y?
decoder_output/mul_1Mul'decoder_output/strided_slice_2:output:0decoder_output/mul_1/y:output:0*
T0*
_output_shapes
: 2
decoder_output/mul_1r
decoder_output/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
decoder_output/stack/3?
decoder_output/stackPack%decoder_output/strided_slice:output:0decoder_output/mul:z:0decoder_output/mul_1:z:0decoder_output/stack/3:output:0*
N*
T0*
_output_shapes
:2
decoder_output/stack?
$decoder_output/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$decoder_output/strided_slice_3/stack?
&decoder_output/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&decoder_output/strided_slice_3/stack_1?
&decoder_output/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&decoder_output/strided_slice_3/stack_2?
decoder_output/strided_slice_3StridedSlicedecoder_output/stack:output:0-decoder_output/strided_slice_3/stack:output:0/decoder_output/strided_slice_3/stack_1:output:0/decoder_output/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
decoder_output/strided_slice_3?
.decoder_output/conv2d_transpose/ReadVariableOpReadVariableOp7decoder_output_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype020
.decoder_output/conv2d_transpose/ReadVariableOp?
decoder_output/conv2d_transposeConv2DBackpropInputdecoder_output/stack:output:06decoder_output/conv2d_transpose/ReadVariableOp:value:0%conv2d_transpose_5/Relu:activations:0*
T0*/
_output_shapes
:?????????  *
paddingSAME*
strides
2!
decoder_output/conv2d_transpose?
%decoder_output/BiasAdd/ReadVariableOpReadVariableOp.decoder_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%decoder_output/BiasAdd/ReadVariableOp?
decoder_output/BiasAddBiasAdd(decoder_output/conv2d_transpose:output:0-decoder_output/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  2
decoder_output/BiasAdd?
decoder_output/SigmoidSigmoiddecoder_output/BiasAdd:output:0*
T0*/
_output_shapes
:?????????  2
decoder_output/Sigmoid?
IdentityIdentitydecoder_output/Sigmoid:y:0*^conv2d_transpose_3/BiasAdd/ReadVariableOp3^conv2d_transpose_3/conv2d_transpose/ReadVariableOp*^conv2d_transpose_4/BiasAdd/ReadVariableOp3^conv2d_transpose_4/conv2d_transpose/ReadVariableOp*^conv2d_transpose_5/BiasAdd/ReadVariableOp3^conv2d_transpose_5/conv2d_transpose/ReadVariableOp&^decoder_output/BiasAdd/ReadVariableOp/^decoder_output/conv2d_transpose/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*
T0*/
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*O
_input_shapes>
<:??????????::::::::::2V
)conv2d_transpose_3/BiasAdd/ReadVariableOp)conv2d_transpose_3/BiasAdd/ReadVariableOp2h
2conv2d_transpose_3/conv2d_transpose/ReadVariableOp2conv2d_transpose_3/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_4/BiasAdd/ReadVariableOp)conv2d_transpose_4/BiasAdd/ReadVariableOp2h
2conv2d_transpose_4/conv2d_transpose/ReadVariableOp2conv2d_transpose_4/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_5/BiasAdd/ReadVariableOp)conv2d_transpose_5/BiasAdd/ReadVariableOp2h
2conv2d_transpose_5/conv2d_transpose/ReadVariableOp2conv2d_transpose_5/conv2d_transpose/ReadVariableOp2N
%decoder_output/BiasAdd/ReadVariableOp%decoder_output/BiasAdd/ReadVariableOp2`
.decoder_output/conv2d_transpose/ReadVariableOp.decoder_output/conv2d_transpose/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
?%
?
B__inference_decoder_layer_call_and_return_conditional_losses_88209

inputs*
&dense_1_statefulpartitionedcall_args_1*
&dense_1_statefulpartitionedcall_args_25
1conv2d_transpose_3_statefulpartitionedcall_args_15
1conv2d_transpose_3_statefulpartitionedcall_args_25
1conv2d_transpose_4_statefulpartitionedcall_args_15
1conv2d_transpose_4_statefulpartitionedcall_args_25
1conv2d_transpose_5_statefulpartitionedcall_args_15
1conv2d_transpose_5_statefulpartitionedcall_args_21
-decoder_output_statefulpartitionedcall_args_11
-decoder_output_statefulpartitionedcall_args_2
identity??*conv2d_transpose_3/StatefulPartitionedCall?*conv2d_transpose_4/StatefulPartitionedCall?*conv2d_transpose_5/StatefulPartitionedCall?&decoder_output/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCallinputs&dense_1_statefulpartitionedcall_args_1&dense_1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:?????????? *-
config_proto

GPU

CPU2*0J 8*K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_880842!
dense_1/StatefulPartitionedCall?
reshape_1/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:??????????*-
config_proto

GPU

CPU2*0J 8*M
fHRF
D__inference_reshape_1_layer_call_and_return_conditional_losses_881102
reshape_1/PartitionedCall?
*conv2d_transpose_3/StatefulPartitionedCallStatefulPartitionedCall"reshape_1/PartitionedCall:output:01conv2d_transpose_3_statefulpartitionedcall_args_11conv2d_transpose_3_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+???????????????????????????*-
config_proto

GPU

CPU2*0J 8*V
fQRO
M__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_879332,
*conv2d_transpose_3/StatefulPartitionedCall?
*conv2d_transpose_4/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_3/StatefulPartitionedCall:output:01conv2d_transpose_4_statefulpartitionedcall_args_11conv2d_transpose_4_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+???????????????????????????*-
config_proto

GPU

CPU2*0J 8*V
fQRO
M__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_879762,
*conv2d_transpose_4/StatefulPartitionedCall?
*conv2d_transpose_5/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_4/StatefulPartitionedCall:output:01conv2d_transpose_5_statefulpartitionedcall_args_11conv2d_transpose_5_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+???????????????????????????*-
config_proto

GPU

CPU2*0J 8*V
fQRO
M__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_880192,
*conv2d_transpose_5/StatefulPartitionedCall?
&decoder_output/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_5/StatefulPartitionedCall:output:0-decoder_output_statefulpartitionedcall_args_1-decoder_output_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+???????????????????????????*-
config_proto

GPU

CPU2*0J 8*R
fMRK
I__inference_decoder_output_layer_call_and_return_conditional_losses_880622(
&decoder_output/StatefulPartitionedCall?
IdentityIdentity/decoder_output/StatefulPartitionedCall:output:0+^conv2d_transpose_3/StatefulPartitionedCall+^conv2d_transpose_4/StatefulPartitionedCall+^conv2d_transpose_5/StatefulPartitionedCall'^decoder_output/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*O
_input_shapes>
<:??????????::::::::::2X
*conv2d_transpose_3/StatefulPartitionedCall*conv2d_transpose_3/StatefulPartitionedCall2X
*conv2d_transpose_4/StatefulPartitionedCall*conv2d_transpose_4/StatefulPartitionedCall2X
*conv2d_transpose_5/StatefulPartitionedCall*conv2d_transpose_5/StatefulPartitionedCall2P
&decoder_output/StatefulPartitionedCall&decoder_output/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:& "
 
_user_specified_nameinputs
?
?
B__inference_encoder_layer_call_and_return_conditional_losses_87857

inputs+
'conv2d_3_statefulpartitionedcall_args_1+
'conv2d_3_statefulpartitionedcall_args_2+
'conv2d_4_statefulpartitionedcall_args_1+
'conv2d_4_statefulpartitionedcall_args_2+
'conv2d_5_statefulpartitionedcall_args_1+
'conv2d_5_statefulpartitionedcall_args_20
,latent_vector_statefulpartitionedcall_args_10
,latent_vector_statefulpartitionedcall_args_2
identity?? conv2d_3/StatefulPartitionedCall? conv2d_4/StatefulPartitionedCall? conv2d_5/StatefulPartitionedCall?%latent_vector/StatefulPartitionedCall?
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCallinputs'conv2d_3_statefulpartitionedcall_args_1'conv2d_3_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:?????????@*-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_conv2d_3_layer_call_and_return_conditional_losses_877202"
 conv2d_3/StatefulPartitionedCall?
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0'conv2d_4_statefulpartitionedcall_args_1'conv2d_4_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:??????????*-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_conv2d_4_layer_call_and_return_conditional_losses_877412"
 conv2d_4/StatefulPartitionedCall?
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0'conv2d_5_statefulpartitionedcall_args_1'conv2d_5_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:??????????*-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_conv2d_5_layer_call_and_return_conditional_losses_877622"
 conv2d_5/StatefulPartitionedCall?
flatten_1/PartitionedCallPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:?????????? *-
config_proto

GPU

CPU2*0J 8*M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_877892
flatten_1/PartitionedCall?
%latent_vector/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0,latent_vector_statefulpartitionedcall_args_1,latent_vector_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????*-
config_proto

GPU

CPU2*0J 8*Q
fLRJ
H__inference_latent_vector_layer_call_and_return_conditional_losses_878072'
%latent_vector/StatefulPartitionedCall?
IdentityIdentity.latent_vector/StatefulPartitionedCall:output:0!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall&^latent_vector/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????  ::::::::2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2N
%latent_vector/StatefulPartitionedCall%latent_vector/StatefulPartitionedCall:& "
 
_user_specified_nameinputs
??
?
B__inference_decoder_layer_call_and_return_conditional_losses_89210

inputs*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource?
;conv2d_transpose_3_conv2d_transpose_readvariableop_resource6
2conv2d_transpose_3_biasadd_readvariableop_resource?
;conv2d_transpose_4_conv2d_transpose_readvariableop_resource6
2conv2d_transpose_4_biasadd_readvariableop_resource?
;conv2d_transpose_5_conv2d_transpose_readvariableop_resource6
2conv2d_transpose_5_biasadd_readvariableop_resource;
7decoder_output_conv2d_transpose_readvariableop_resource2
.decoder_output_biasadd_readvariableop_resource
identity??)conv2d_transpose_3/BiasAdd/ReadVariableOp?2conv2d_transpose_3/conv2d_transpose/ReadVariableOp?)conv2d_transpose_4/BiasAdd/ReadVariableOp?2conv2d_transpose_4/conv2d_transpose/ReadVariableOp?)conv2d_transpose_5/BiasAdd/ReadVariableOp?2conv2d_transpose_5/conv2d_transpose/ReadVariableOp?%decoder_output/BiasAdd/ReadVariableOp?.decoder_output/conv2d_transpose/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
?? *
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMulinputs%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????? 2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:? *
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????? 2
dense_1/BiasAddj
reshape_1/ShapeShapedense_1/BiasAdd:output:0*
T0*
_output_shapes
:2
reshape_1/Shape?
reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_1/strided_slice/stack?
reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_1/strided_slice/stack_1?
reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_1/strided_slice/stack_2?
reshape_1/strided_sliceStridedSlicereshape_1/Shape:output:0&reshape_1/strided_slice/stack:output:0(reshape_1/strided_slice/stack_1:output:0(reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_1/strided_slicex
reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_1/Reshape/shape/1x
reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_1/Reshape/shape/2y
reshape_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :?2
reshape_1/Reshape/shape/3?
reshape_1/Reshape/shapePack reshape_1/strided_slice:output:0"reshape_1/Reshape/shape/1:output:0"reshape_1/Reshape/shape/2:output:0"reshape_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape_1/Reshape/shape?
reshape_1/ReshapeReshapedense_1/BiasAdd:output:0 reshape_1/Reshape/shape:output:0*
T0*0
_output_shapes
:??????????2
reshape_1/Reshape~
conv2d_transpose_3/ShapeShapereshape_1/Reshape:output:0*
T0*
_output_shapes
:2
conv2d_transpose_3/Shape?
&conv2d_transpose_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose_3/strided_slice/stack?
(conv2d_transpose_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_3/strided_slice/stack_1?
(conv2d_transpose_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_3/strided_slice/stack_2?
 conv2d_transpose_3/strided_sliceStridedSlice!conv2d_transpose_3/Shape:output:0/conv2d_transpose_3/strided_slice/stack:output:01conv2d_transpose_3/strided_slice/stack_1:output:01conv2d_transpose_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose_3/strided_slice?
(conv2d_transpose_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_3/strided_slice_1/stack?
*conv2d_transpose_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_3/strided_slice_1/stack_1?
*conv2d_transpose_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_3/strided_slice_1/stack_2?
"conv2d_transpose_3/strided_slice_1StridedSlice!conv2d_transpose_3/Shape:output:01conv2d_transpose_3/strided_slice_1/stack:output:03conv2d_transpose_3/strided_slice_1/stack_1:output:03conv2d_transpose_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_3/strided_slice_1?
(conv2d_transpose_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_3/strided_slice_2/stack?
*conv2d_transpose_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_3/strided_slice_2/stack_1?
*conv2d_transpose_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_3/strided_slice_2/stack_2?
"conv2d_transpose_3/strided_slice_2StridedSlice!conv2d_transpose_3/Shape:output:01conv2d_transpose_3/strided_slice_2/stack:output:03conv2d_transpose_3/strided_slice_2/stack_1:output:03conv2d_transpose_3/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_3/strided_slice_2v
conv2d_transpose_3/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_3/mul/y?
conv2d_transpose_3/mulMul+conv2d_transpose_3/strided_slice_1:output:0!conv2d_transpose_3/mul/y:output:0*
T0*
_output_shapes
: 2
conv2d_transpose_3/mulz
conv2d_transpose_3/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_3/mul_1/y?
conv2d_transpose_3/mul_1Mul+conv2d_transpose_3/strided_slice_2:output:0#conv2d_transpose_3/mul_1/y:output:0*
T0*
_output_shapes
: 2
conv2d_transpose_3/mul_1z
conv2d_transpose_3/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_3/stack/3?
conv2d_transpose_3/stackPack)conv2d_transpose_3/strided_slice:output:0conv2d_transpose_3/mul:z:0conv2d_transpose_3/mul_1:z:0#conv2d_transpose_3/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_3/stack?
(conv2d_transpose_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(conv2d_transpose_3/strided_slice_3/stack?
*conv2d_transpose_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_3/strided_slice_3/stack_1?
*conv2d_transpose_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_3/strided_slice_3/stack_2?
"conv2d_transpose_3/strided_slice_3StridedSlice!conv2d_transpose_3/stack:output:01conv2d_transpose_3/strided_slice_3/stack:output:03conv2d_transpose_3/strided_slice_3/stack_1:output:03conv2d_transpose_3/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_3/strided_slice_3?
2conv2d_transpose_3/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_3_conv2d_transpose_readvariableop_resource*'
_output_shapes
:?*
dtype024
2conv2d_transpose_3/conv2d_transpose/ReadVariableOp?
#conv2d_transpose_3/conv2d_transposeConv2DBackpropInput!conv2d_transpose_3/stack:output:0:conv2d_transpose_3/conv2d_transpose/ReadVariableOp:value:0reshape_1/Reshape:output:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2%
#conv2d_transpose_3/conv2d_transpose?
)conv2d_transpose_3/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)conv2d_transpose_3/BiasAdd/ReadVariableOp?
conv2d_transpose_3/BiasAddBiasAdd,conv2d_transpose_3/conv2d_transpose:output:01conv2d_transpose_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d_transpose_3/BiasAdd?
conv2d_transpose_3/ReluRelu#conv2d_transpose_3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
conv2d_transpose_3/Relu?
conv2d_transpose_4/ShapeShape%conv2d_transpose_3/Relu:activations:0*
T0*
_output_shapes
:2
conv2d_transpose_4/Shape?
&conv2d_transpose_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose_4/strided_slice/stack?
(conv2d_transpose_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_4/strided_slice/stack_1?
(conv2d_transpose_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_4/strided_slice/stack_2?
 conv2d_transpose_4/strided_sliceStridedSlice!conv2d_transpose_4/Shape:output:0/conv2d_transpose_4/strided_slice/stack:output:01conv2d_transpose_4/strided_slice/stack_1:output:01conv2d_transpose_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose_4/strided_slice?
(conv2d_transpose_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_4/strided_slice_1/stack?
*conv2d_transpose_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_4/strided_slice_1/stack_1?
*conv2d_transpose_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_4/strided_slice_1/stack_2?
"conv2d_transpose_4/strided_slice_1StridedSlice!conv2d_transpose_4/Shape:output:01conv2d_transpose_4/strided_slice_1/stack:output:03conv2d_transpose_4/strided_slice_1/stack_1:output:03conv2d_transpose_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_4/strided_slice_1?
(conv2d_transpose_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_4/strided_slice_2/stack?
*conv2d_transpose_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_4/strided_slice_2/stack_1?
*conv2d_transpose_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_4/strided_slice_2/stack_2?
"conv2d_transpose_4/strided_slice_2StridedSlice!conv2d_transpose_4/Shape:output:01conv2d_transpose_4/strided_slice_2/stack:output:03conv2d_transpose_4/strided_slice_2/stack_1:output:03conv2d_transpose_4/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_4/strided_slice_2v
conv2d_transpose_4/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_4/mul/y?
conv2d_transpose_4/mulMul+conv2d_transpose_4/strided_slice_1:output:0!conv2d_transpose_4/mul/y:output:0*
T0*
_output_shapes
: 2
conv2d_transpose_4/mulz
conv2d_transpose_4/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_4/mul_1/y?
conv2d_transpose_4/mul_1Mul+conv2d_transpose_4/strided_slice_2:output:0#conv2d_transpose_4/mul_1/y:output:0*
T0*
_output_shapes
: 2
conv2d_transpose_4/mul_1z
conv2d_transpose_4/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_4/stack/3?
conv2d_transpose_4/stackPack)conv2d_transpose_4/strided_slice:output:0conv2d_transpose_4/mul:z:0conv2d_transpose_4/mul_1:z:0#conv2d_transpose_4/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_4/stack?
(conv2d_transpose_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(conv2d_transpose_4/strided_slice_3/stack?
*conv2d_transpose_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_4/strided_slice_3/stack_1?
*conv2d_transpose_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_4/strided_slice_3/stack_2?
"conv2d_transpose_4/strided_slice_3StridedSlice!conv2d_transpose_4/stack:output:01conv2d_transpose_4/strided_slice_3/stack:output:03conv2d_transpose_4/strided_slice_3/stack_1:output:03conv2d_transpose_4/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_4/strided_slice_3?
2conv2d_transpose_4/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_4_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype024
2conv2d_transpose_4/conv2d_transpose/ReadVariableOp?
#conv2d_transpose_4/conv2d_transposeConv2DBackpropInput!conv2d_transpose_4/stack:output:0:conv2d_transpose_4/conv2d_transpose/ReadVariableOp:value:0%conv2d_transpose_3/Relu:activations:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2%
#conv2d_transpose_4/conv2d_transpose?
)conv2d_transpose_4/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)conv2d_transpose_4/BiasAdd/ReadVariableOp?
conv2d_transpose_4/BiasAddBiasAdd,conv2d_transpose_4/conv2d_transpose:output:01conv2d_transpose_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d_transpose_4/BiasAdd?
conv2d_transpose_4/ReluRelu#conv2d_transpose_4/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
conv2d_transpose_4/Relu?
conv2d_transpose_5/ShapeShape%conv2d_transpose_4/Relu:activations:0*
T0*
_output_shapes
:2
conv2d_transpose_5/Shape?
&conv2d_transpose_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose_5/strided_slice/stack?
(conv2d_transpose_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_5/strided_slice/stack_1?
(conv2d_transpose_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_5/strided_slice/stack_2?
 conv2d_transpose_5/strided_sliceStridedSlice!conv2d_transpose_5/Shape:output:0/conv2d_transpose_5/strided_slice/stack:output:01conv2d_transpose_5/strided_slice/stack_1:output:01conv2d_transpose_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose_5/strided_slice?
(conv2d_transpose_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_5/strided_slice_1/stack?
*conv2d_transpose_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_5/strided_slice_1/stack_1?
*conv2d_transpose_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_5/strided_slice_1/stack_2?
"conv2d_transpose_5/strided_slice_1StridedSlice!conv2d_transpose_5/Shape:output:01conv2d_transpose_5/strided_slice_1/stack:output:03conv2d_transpose_5/strided_slice_1/stack_1:output:03conv2d_transpose_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_5/strided_slice_1?
(conv2d_transpose_5/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_5/strided_slice_2/stack?
*conv2d_transpose_5/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_5/strided_slice_2/stack_1?
*conv2d_transpose_5/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_5/strided_slice_2/stack_2?
"conv2d_transpose_5/strided_slice_2StridedSlice!conv2d_transpose_5/Shape:output:01conv2d_transpose_5/strided_slice_2/stack:output:03conv2d_transpose_5/strided_slice_2/stack_1:output:03conv2d_transpose_5/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_5/strided_slice_2v
conv2d_transpose_5/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_5/mul/y?
conv2d_transpose_5/mulMul+conv2d_transpose_5/strided_slice_1:output:0!conv2d_transpose_5/mul/y:output:0*
T0*
_output_shapes
: 2
conv2d_transpose_5/mulz
conv2d_transpose_5/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_5/mul_1/y?
conv2d_transpose_5/mul_1Mul+conv2d_transpose_5/strided_slice_2:output:0#conv2d_transpose_5/mul_1/y:output:0*
T0*
_output_shapes
: 2
conv2d_transpose_5/mul_1z
conv2d_transpose_5/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_5/stack/3?
conv2d_transpose_5/stackPack)conv2d_transpose_5/strided_slice:output:0conv2d_transpose_5/mul:z:0conv2d_transpose_5/mul_1:z:0#conv2d_transpose_5/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_5/stack?
(conv2d_transpose_5/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(conv2d_transpose_5/strided_slice_3/stack?
*conv2d_transpose_5/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_5/strided_slice_3/stack_1?
*conv2d_transpose_5/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_5/strided_slice_3/stack_2?
"conv2d_transpose_5/strided_slice_3StridedSlice!conv2d_transpose_5/stack:output:01conv2d_transpose_5/strided_slice_3/stack:output:03conv2d_transpose_5/strided_slice_3/stack_1:output:03conv2d_transpose_5/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_5/strided_slice_3?
2conv2d_transpose_5/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_5_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype024
2conv2d_transpose_5/conv2d_transpose/ReadVariableOp?
#conv2d_transpose_5/conv2d_transposeConv2DBackpropInput!conv2d_transpose_5/stack:output:0:conv2d_transpose_5/conv2d_transpose/ReadVariableOp:value:0%conv2d_transpose_4/Relu:activations:0*
T0*/
_output_shapes
:?????????  *
paddingSAME*
strides
2%
#conv2d_transpose_5/conv2d_transpose?
)conv2d_transpose_5/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)conv2d_transpose_5/BiasAdd/ReadVariableOp?
conv2d_transpose_5/BiasAddBiasAdd,conv2d_transpose_5/conv2d_transpose:output:01conv2d_transpose_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  2
conv2d_transpose_5/BiasAdd?
conv2d_transpose_5/ReluRelu#conv2d_transpose_5/BiasAdd:output:0*
T0*/
_output_shapes
:?????????  2
conv2d_transpose_5/Relu?
decoder_output/ShapeShape%conv2d_transpose_5/Relu:activations:0*
T0*
_output_shapes
:2
decoder_output/Shape?
"decoder_output/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"decoder_output/strided_slice/stack?
$decoder_output/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$decoder_output/strided_slice/stack_1?
$decoder_output/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$decoder_output/strided_slice/stack_2?
decoder_output/strided_sliceStridedSlicedecoder_output/Shape:output:0+decoder_output/strided_slice/stack:output:0-decoder_output/strided_slice/stack_1:output:0-decoder_output/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
decoder_output/strided_slice?
$decoder_output/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2&
$decoder_output/strided_slice_1/stack?
&decoder_output/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&decoder_output/strided_slice_1/stack_1?
&decoder_output/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&decoder_output/strided_slice_1/stack_2?
decoder_output/strided_slice_1StridedSlicedecoder_output/Shape:output:0-decoder_output/strided_slice_1/stack:output:0/decoder_output/strided_slice_1/stack_1:output:0/decoder_output/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
decoder_output/strided_slice_1?
$decoder_output/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2&
$decoder_output/strided_slice_2/stack?
&decoder_output/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&decoder_output/strided_slice_2/stack_1?
&decoder_output/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&decoder_output/strided_slice_2/stack_2?
decoder_output/strided_slice_2StridedSlicedecoder_output/Shape:output:0-decoder_output/strided_slice_2/stack:output:0/decoder_output/strided_slice_2/stack_1:output:0/decoder_output/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
decoder_output/strided_slice_2n
decoder_output/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
decoder_output/mul/y?
decoder_output/mulMul'decoder_output/strided_slice_1:output:0decoder_output/mul/y:output:0*
T0*
_output_shapes
: 2
decoder_output/mulr
decoder_output/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
decoder_output/mul_1/y?
decoder_output/mul_1Mul'decoder_output/strided_slice_2:output:0decoder_output/mul_1/y:output:0*
T0*
_output_shapes
: 2
decoder_output/mul_1r
decoder_output/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
decoder_output/stack/3?
decoder_output/stackPack%decoder_output/strided_slice:output:0decoder_output/mul:z:0decoder_output/mul_1:z:0decoder_output/stack/3:output:0*
N*
T0*
_output_shapes
:2
decoder_output/stack?
$decoder_output/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$decoder_output/strided_slice_3/stack?
&decoder_output/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&decoder_output/strided_slice_3/stack_1?
&decoder_output/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&decoder_output/strided_slice_3/stack_2?
decoder_output/strided_slice_3StridedSlicedecoder_output/stack:output:0-decoder_output/strided_slice_3/stack:output:0/decoder_output/strided_slice_3/stack_1:output:0/decoder_output/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
decoder_output/strided_slice_3?
.decoder_output/conv2d_transpose/ReadVariableOpReadVariableOp7decoder_output_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype020
.decoder_output/conv2d_transpose/ReadVariableOp?
decoder_output/conv2d_transposeConv2DBackpropInputdecoder_output/stack:output:06decoder_output/conv2d_transpose/ReadVariableOp:value:0%conv2d_transpose_5/Relu:activations:0*
T0*/
_output_shapes
:?????????  *
paddingSAME*
strides
2!
decoder_output/conv2d_transpose?
%decoder_output/BiasAdd/ReadVariableOpReadVariableOp.decoder_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%decoder_output/BiasAdd/ReadVariableOp?
decoder_output/BiasAddBiasAdd(decoder_output/conv2d_transpose:output:0-decoder_output/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  2
decoder_output/BiasAdd?
decoder_output/SigmoidSigmoiddecoder_output/BiasAdd:output:0*
T0*/
_output_shapes
:?????????  2
decoder_output/Sigmoid?
IdentityIdentitydecoder_output/Sigmoid:y:0*^conv2d_transpose_3/BiasAdd/ReadVariableOp3^conv2d_transpose_3/conv2d_transpose/ReadVariableOp*^conv2d_transpose_4/BiasAdd/ReadVariableOp3^conv2d_transpose_4/conv2d_transpose/ReadVariableOp*^conv2d_transpose_5/BiasAdd/ReadVariableOp3^conv2d_transpose_5/conv2d_transpose/ReadVariableOp&^decoder_output/BiasAdd/ReadVariableOp/^decoder_output/conv2d_transpose/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*
T0*/
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*O
_input_shapes>
<:??????????::::::::::2V
)conv2d_transpose_3/BiasAdd/ReadVariableOp)conv2d_transpose_3/BiasAdd/ReadVariableOp2h
2conv2d_transpose_3/conv2d_transpose/ReadVariableOp2conv2d_transpose_3/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_4/BiasAdd/ReadVariableOp)conv2d_transpose_4/BiasAdd/ReadVariableOp2h
2conv2d_transpose_4/conv2d_transpose/ReadVariableOp2conv2d_transpose_4/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_5/BiasAdd/ReadVariableOp)conv2d_transpose_5/BiasAdd/ReadVariableOp2h
2conv2d_transpose_5/conv2d_transpose/ReadVariableOp2conv2d_transpose_5/conv2d_transpose/ReadVariableOp2N
%decoder_output/BiasAdd/ReadVariableOp%decoder_output/BiasAdd/ReadVariableOp2`
.decoder_output/conv2d_transpose/ReadVariableOp.decoder_output/conv2d_transpose/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
?

?
'__inference_encoder_layer_call_fn_87898
encoder_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallencoder_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8*
Tin
2	*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????*-
config_proto

GPU

CPU2*0J 8*K
fFRD
B__inference_encoder_layer_call_and_return_conditional_losses_878872
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????  ::::::::22
StatefulPartitionedCallStatefulPartitionedCall:- )
'
_user_specified_nameencoder_input
?
?
F__inference_autoencoder_layer_call_and_return_conditional_losses_88354

inputs*
&encoder_statefulpartitionedcall_args_1*
&encoder_statefulpartitionedcall_args_2*
&encoder_statefulpartitionedcall_args_3*
&encoder_statefulpartitionedcall_args_4*
&encoder_statefulpartitionedcall_args_5*
&encoder_statefulpartitionedcall_args_6*
&encoder_statefulpartitionedcall_args_7*
&encoder_statefulpartitionedcall_args_8*
&decoder_statefulpartitionedcall_args_1*
&decoder_statefulpartitionedcall_args_2*
&decoder_statefulpartitionedcall_args_3*
&decoder_statefulpartitionedcall_args_4*
&decoder_statefulpartitionedcall_args_5*
&decoder_statefulpartitionedcall_args_6*
&decoder_statefulpartitionedcall_args_7*
&decoder_statefulpartitionedcall_args_8*
&decoder_statefulpartitionedcall_args_9+
'decoder_statefulpartitionedcall_args_10
identity??decoder/StatefulPartitionedCall?encoder/StatefulPartitionedCall?
encoder/StatefulPartitionedCallStatefulPartitionedCallinputs&encoder_statefulpartitionedcall_args_1&encoder_statefulpartitionedcall_args_2&encoder_statefulpartitionedcall_args_3&encoder_statefulpartitionedcall_args_4&encoder_statefulpartitionedcall_args_5&encoder_statefulpartitionedcall_args_6&encoder_statefulpartitionedcall_args_7&encoder_statefulpartitionedcall_args_8*
Tin
2	*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????*-
config_proto

GPU

CPU2*0J 8*K
fFRD
B__inference_encoder_layer_call_and_return_conditional_losses_878572!
encoder/StatefulPartitionedCall?
decoder/StatefulPartitionedCallStatefulPartitionedCall(encoder/StatefulPartitionedCall:output:0&decoder_statefulpartitionedcall_args_1&decoder_statefulpartitionedcall_args_2&decoder_statefulpartitionedcall_args_3&decoder_statefulpartitionedcall_args_4&decoder_statefulpartitionedcall_args_5&decoder_statefulpartitionedcall_args_6&decoder_statefulpartitionedcall_args_7&decoder_statefulpartitionedcall_args_8&decoder_statefulpartitionedcall_args_9'decoder_statefulpartitionedcall_args_10*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+???????????????????????????*-
config_proto

GPU

CPU2*0J 8*K
fFRD
B__inference_decoder_layer_call_and_return_conditional_losses_881742!
decoder/StatefulPartitionedCall?
IdentityIdentity(decoder/StatefulPartitionedCall:output:0 ^decoder/StatefulPartitionedCall ^encoder/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*v
_input_shapese
c:?????????  ::::::::::::::::::2B
decoder/StatefulPartitionedCalldecoder/StatefulPartitionedCall2B
encoder/StatefulPartitionedCallencoder/StatefulPartitionedCall:& "
 
_user_specified_nameinputs
?
?
F__inference_autoencoder_layer_call_and_return_conditional_losses_88303
encoder_input*
&encoder_statefulpartitionedcall_args_1*
&encoder_statefulpartitionedcall_args_2*
&encoder_statefulpartitionedcall_args_3*
&encoder_statefulpartitionedcall_args_4*
&encoder_statefulpartitionedcall_args_5*
&encoder_statefulpartitionedcall_args_6*
&encoder_statefulpartitionedcall_args_7*
&encoder_statefulpartitionedcall_args_8*
&decoder_statefulpartitionedcall_args_1*
&decoder_statefulpartitionedcall_args_2*
&decoder_statefulpartitionedcall_args_3*
&decoder_statefulpartitionedcall_args_4*
&decoder_statefulpartitionedcall_args_5*
&decoder_statefulpartitionedcall_args_6*
&decoder_statefulpartitionedcall_args_7*
&decoder_statefulpartitionedcall_args_8*
&decoder_statefulpartitionedcall_args_9+
'decoder_statefulpartitionedcall_args_10
identity??decoder/StatefulPartitionedCall?encoder/StatefulPartitionedCall?
encoder/StatefulPartitionedCallStatefulPartitionedCallencoder_input&encoder_statefulpartitionedcall_args_1&encoder_statefulpartitionedcall_args_2&encoder_statefulpartitionedcall_args_3&encoder_statefulpartitionedcall_args_4&encoder_statefulpartitionedcall_args_5&encoder_statefulpartitionedcall_args_6&encoder_statefulpartitionedcall_args_7&encoder_statefulpartitionedcall_args_8*
Tin
2	*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????*-
config_proto

GPU

CPU2*0J 8*K
fFRD
B__inference_encoder_layer_call_and_return_conditional_losses_878572!
encoder/StatefulPartitionedCall?
decoder/StatefulPartitionedCallStatefulPartitionedCall(encoder/StatefulPartitionedCall:output:0&decoder_statefulpartitionedcall_args_1&decoder_statefulpartitionedcall_args_2&decoder_statefulpartitionedcall_args_3&decoder_statefulpartitionedcall_args_4&decoder_statefulpartitionedcall_args_5&decoder_statefulpartitionedcall_args_6&decoder_statefulpartitionedcall_args_7&decoder_statefulpartitionedcall_args_8&decoder_statefulpartitionedcall_args_9'decoder_statefulpartitionedcall_args_10*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+???????????????????????????*-
config_proto

GPU

CPU2*0J 8*K
fFRD
B__inference_decoder_layer_call_and_return_conditional_losses_881742!
decoder/StatefulPartitionedCall?
IdentityIdentity(decoder/StatefulPartitionedCall:output:0 ^decoder/StatefulPartitionedCall ^encoder/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*v
_input_shapese
c:?????????  ::::::::::::::::::2B
decoder/StatefulPartitionedCalldecoder/StatefulPartitionedCall2B
encoder/StatefulPartitionedCallencoder/StatefulPartitionedCall:- )
'
_user_specified_nameencoder_input
?
?
F__inference_autoencoder_layer_call_and_return_conditional_losses_88401

inputs*
&encoder_statefulpartitionedcall_args_1*
&encoder_statefulpartitionedcall_args_2*
&encoder_statefulpartitionedcall_args_3*
&encoder_statefulpartitionedcall_args_4*
&encoder_statefulpartitionedcall_args_5*
&encoder_statefulpartitionedcall_args_6*
&encoder_statefulpartitionedcall_args_7*
&encoder_statefulpartitionedcall_args_8*
&decoder_statefulpartitionedcall_args_1*
&decoder_statefulpartitionedcall_args_2*
&decoder_statefulpartitionedcall_args_3*
&decoder_statefulpartitionedcall_args_4*
&decoder_statefulpartitionedcall_args_5*
&decoder_statefulpartitionedcall_args_6*
&decoder_statefulpartitionedcall_args_7*
&decoder_statefulpartitionedcall_args_8*
&decoder_statefulpartitionedcall_args_9+
'decoder_statefulpartitionedcall_args_10
identity??decoder/StatefulPartitionedCall?encoder/StatefulPartitionedCall?
encoder/StatefulPartitionedCallStatefulPartitionedCallinputs&encoder_statefulpartitionedcall_args_1&encoder_statefulpartitionedcall_args_2&encoder_statefulpartitionedcall_args_3&encoder_statefulpartitionedcall_args_4&encoder_statefulpartitionedcall_args_5&encoder_statefulpartitionedcall_args_6&encoder_statefulpartitionedcall_args_7&encoder_statefulpartitionedcall_args_8*
Tin
2	*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????*-
config_proto

GPU

CPU2*0J 8*K
fFRD
B__inference_encoder_layer_call_and_return_conditional_losses_878872!
encoder/StatefulPartitionedCall?
decoder/StatefulPartitionedCallStatefulPartitionedCall(encoder/StatefulPartitionedCall:output:0&decoder_statefulpartitionedcall_args_1&decoder_statefulpartitionedcall_args_2&decoder_statefulpartitionedcall_args_3&decoder_statefulpartitionedcall_args_4&decoder_statefulpartitionedcall_args_5&decoder_statefulpartitionedcall_args_6&decoder_statefulpartitionedcall_args_7&decoder_statefulpartitionedcall_args_8&decoder_statefulpartitionedcall_args_9'decoder_statefulpartitionedcall_args_10*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+???????????????????????????*-
config_proto

GPU

CPU2*0J 8*K
fFRD
B__inference_decoder_layer_call_and_return_conditional_losses_882092!
decoder/StatefulPartitionedCall?
IdentityIdentity(decoder/StatefulPartitionedCall:output:0 ^decoder/StatefulPartitionedCall ^encoder/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*v
_input_shapese
c:?????????  ::::::::::::::::::2B
decoder/StatefulPartitionedCalldecoder/StatefulPartitionedCall2B
encoder/StatefulPartitionedCallencoder/StatefulPartitionedCall:& "
 
_user_specified_nameinputs
?
?
(__inference_conv2d_5_layer_call_fn_87770

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,????????????????????????????*-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_conv2d_5_layer_call_and_return_conditional_losses_877622
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,????????????????????????????::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?*
?
B__inference_encoder_layer_call_and_return_conditional_losses_88904

inputs+
'conv2d_3_conv2d_readvariableop_resource,
(conv2d_3_biasadd_readvariableop_resource+
'conv2d_4_conv2d_readvariableop_resource,
(conv2d_4_biasadd_readvariableop_resource+
'conv2d_5_conv2d_readvariableop_resource,
(conv2d_5_biasadd_readvariableop_resource0
,latent_vector_matmul_readvariableop_resource1
-latent_vector_biasadd_readvariableop_resource
identity??conv2d_3/BiasAdd/ReadVariableOp?conv2d_3/Conv2D/ReadVariableOp?conv2d_4/BiasAdd/ReadVariableOp?conv2d_4/Conv2D/ReadVariableOp?conv2d_5/BiasAdd/ReadVariableOp?conv2d_5/Conv2D/ReadVariableOp?$latent_vector/BiasAdd/ReadVariableOp?#latent_vector/MatMul/ReadVariableOp?
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02 
conv2d_3/Conv2D/ReadVariableOp?
conv2d_3/Conv2DConv2Dinputs&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
conv2d_3/Conv2D?
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_3/BiasAdd/ReadVariableOp?
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_3/BiasAdd{
conv2d_3/ReluReluconv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
conv2d_3/Relu?
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02 
conv2d_4/Conv2D/ReadVariableOp?
conv2d_4/Conv2DConv2Dconv2d_3/Relu:activations:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv2d_4/Conv2D?
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
conv2d_4/BiasAdd/ReadVariableOp?
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_4/BiasAdd|
conv2d_4/ReluReluconv2d_4/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
conv2d_4/Relu?
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02 
conv2d_5/Conv2D/ReadVariableOp?
conv2d_5/Conv2DConv2Dconv2d_4/Relu:activations:0&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv2d_5/Conv2D?
conv2d_5/BiasAdd/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
conv2d_5/BiasAdd/ReadVariableOp?
conv2d_5/BiasAddBiasAddconv2d_5/Conv2D:output:0'conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_5/BiasAdd|
conv2d_5/ReluReluconv2d_5/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
conv2d_5/Relus
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_1/Const?
flatten_1/ReshapeReshapeconv2d_5/Relu:activations:0flatten_1/Const:output:0*
T0*(
_output_shapes
:?????????? 2
flatten_1/Reshape?
#latent_vector/MatMul/ReadVariableOpReadVariableOp,latent_vector_matmul_readvariableop_resource* 
_output_shapes
:
? ?*
dtype02%
#latent_vector/MatMul/ReadVariableOp?
latent_vector/MatMulMatMulflatten_1/Reshape:output:0+latent_vector/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
latent_vector/MatMul?
$latent_vector/BiasAdd/ReadVariableOpReadVariableOp-latent_vector_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$latent_vector/BiasAdd/ReadVariableOp?
latent_vector/BiasAddBiasAddlatent_vector/MatMul:product:0,latent_vector/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
latent_vector/BiasAdd?
IdentityIdentitylatent_vector/BiasAdd:output:0 ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp ^conv2d_5/BiasAdd/ReadVariableOp^conv2d_5/Conv2D/ReadVariableOp%^latent_vector/BiasAdd/ReadVariableOp$^latent_vector/MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????  ::::::::2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2B
conv2d_4/BiasAdd/ReadVariableOpconv2d_4/BiasAdd/ReadVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp2B
conv2d_5/BiasAdd/ReadVariableOpconv2d_5/BiasAdd/ReadVariableOp2@
conv2d_5/Conv2D/ReadVariableOpconv2d_5/Conv2D/ReadVariableOp2L
$latent_vector/BiasAdd/ReadVariableOp$latent_vector/BiasAdd/ReadVariableOp2J
#latent_vector/MatMul/ReadVariableOp#latent_vector/MatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
?
E
)__inference_flatten_1_layer_call_fn_89251

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:?????????? *-
config_proto

GPU

CPU2*0J 8*M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_877892
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:?????????? 2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:& "
 
_user_specified_nameinputs
?
?
-__inference_latent_vector_layer_call_fn_89268

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????*-
config_proto

GPU

CPU2*0J 8*Q
fLRJ
H__inference_latent_vector_layer_call_and_return_conditional_losses_878072
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????? ::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?$
?
M__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_87933

inputs,
(conv2d_transpose_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/3?
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*'
_output_shapes
:?*
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
2
conv2d_transpose?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,????????????????????????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:& "
 
_user_specified_nameinputs
?
?
H__inference_latent_vector_layer_call_and_return_conditional_losses_87807

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
? ?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
?
?
B__inference_dense_1_layer_call_and_return_conditional_losses_89278

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
?? *
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????? 2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:? *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????? 2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:?????????? 2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
?
`
D__inference_flatten_1_layer_call_and_return_conditional_losses_89246

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:?????????? 2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:?????????? 2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:& "
 
_user_specified_nameinputs
?
?
2__inference_conv2d_transpose_4_layer_call_fn_87984

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+???????????????????????????*-
config_proto

GPU

CPU2*0J 8*V
fQRO
M__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_879762
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?
?
#__inference_signature_wrapper_88454
encoder_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallencoder_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:?????????  *-
config_proto

GPU

CPU2*0J 8*)
f$R"
 __inference__wrapped_model_877072
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*v
_input_shapese
c:?????????  ::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:- )
'
_user_specified_nameencoder_input
?
?
B__inference_encoder_layer_call_and_return_conditional_losses_87820
encoder_input+
'conv2d_3_statefulpartitionedcall_args_1+
'conv2d_3_statefulpartitionedcall_args_2+
'conv2d_4_statefulpartitionedcall_args_1+
'conv2d_4_statefulpartitionedcall_args_2+
'conv2d_5_statefulpartitionedcall_args_1+
'conv2d_5_statefulpartitionedcall_args_20
,latent_vector_statefulpartitionedcall_args_10
,latent_vector_statefulpartitionedcall_args_2
identity?? conv2d_3/StatefulPartitionedCall? conv2d_4/StatefulPartitionedCall? conv2d_5/StatefulPartitionedCall?%latent_vector/StatefulPartitionedCall?
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCallencoder_input'conv2d_3_statefulpartitionedcall_args_1'conv2d_3_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:?????????@*-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_conv2d_3_layer_call_and_return_conditional_losses_877202"
 conv2d_3/StatefulPartitionedCall?
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0'conv2d_4_statefulpartitionedcall_args_1'conv2d_4_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:??????????*-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_conv2d_4_layer_call_and_return_conditional_losses_877412"
 conv2d_4/StatefulPartitionedCall?
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0'conv2d_5_statefulpartitionedcall_args_1'conv2d_5_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:??????????*-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_conv2d_5_layer_call_and_return_conditional_losses_877622"
 conv2d_5/StatefulPartitionedCall?
flatten_1/PartitionedCallPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:?????????? *-
config_proto

GPU

CPU2*0J 8*M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_877892
flatten_1/PartitionedCall?
%latent_vector/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0,latent_vector_statefulpartitionedcall_args_1,latent_vector_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????*-
config_proto

GPU

CPU2*0J 8*Q
fLRJ
H__inference_latent_vector_layer_call_and_return_conditional_losses_878072'
%latent_vector/StatefulPartitionedCall?
IdentityIdentity.latent_vector/StatefulPartitionedCall:output:0!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall&^latent_vector/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????  ::::::::2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2N
%latent_vector/StatefulPartitionedCall%latent_vector/StatefulPartitionedCall:- )
'
_user_specified_nameencoder_input
?
?
C__inference_conv2d_3_layer_call_and_return_conditional_losses_87720

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOpo
dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
dilation_rate?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:& "
 
_user_specified_nameinputs
?

?
'__inference_encoder_layer_call_fn_87868
encoder_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallencoder_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8*
Tin
2	*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????*-
config_proto

GPU

CPU2*0J 8*K
fFRD
B__inference_encoder_layer_call_and_return_conditional_losses_878572
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????  ::::::::22
StatefulPartitionedCallStatefulPartitionedCall:- )
'
_user_specified_nameencoder_input
?
?
.__inference_decoder_output_layer_call_fn_88070

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+???????????????????????????*-
config_proto

GPU

CPU2*0J 8*R
fMRK
I__inference_decoder_output_layer_call_and_return_conditional_losses_880622
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
??
?
F__inference_autoencoder_layer_call_and_return_conditional_losses_88623

inputs3
/encoder_conv2d_3_conv2d_readvariableop_resource4
0encoder_conv2d_3_biasadd_readvariableop_resource3
/encoder_conv2d_4_conv2d_readvariableop_resource4
0encoder_conv2d_4_biasadd_readvariableop_resource3
/encoder_conv2d_5_conv2d_readvariableop_resource4
0encoder_conv2d_5_biasadd_readvariableop_resource8
4encoder_latent_vector_matmul_readvariableop_resource9
5encoder_latent_vector_biasadd_readvariableop_resource2
.decoder_dense_1_matmul_readvariableop_resource3
/decoder_dense_1_biasadd_readvariableop_resourceG
Cdecoder_conv2d_transpose_3_conv2d_transpose_readvariableop_resource>
:decoder_conv2d_transpose_3_biasadd_readvariableop_resourceG
Cdecoder_conv2d_transpose_4_conv2d_transpose_readvariableop_resource>
:decoder_conv2d_transpose_4_biasadd_readvariableop_resourceG
Cdecoder_conv2d_transpose_5_conv2d_transpose_readvariableop_resource>
:decoder_conv2d_transpose_5_biasadd_readvariableop_resourceC
?decoder_decoder_output_conv2d_transpose_readvariableop_resource:
6decoder_decoder_output_biasadd_readvariableop_resource
identity??1decoder/conv2d_transpose_3/BiasAdd/ReadVariableOp?:decoder/conv2d_transpose_3/conv2d_transpose/ReadVariableOp?1decoder/conv2d_transpose_4/BiasAdd/ReadVariableOp?:decoder/conv2d_transpose_4/conv2d_transpose/ReadVariableOp?1decoder/conv2d_transpose_5/BiasAdd/ReadVariableOp?:decoder/conv2d_transpose_5/conv2d_transpose/ReadVariableOp?-decoder/decoder_output/BiasAdd/ReadVariableOp?6decoder/decoder_output/conv2d_transpose/ReadVariableOp?&decoder/dense_1/BiasAdd/ReadVariableOp?%decoder/dense_1/MatMul/ReadVariableOp?'encoder/conv2d_3/BiasAdd/ReadVariableOp?&encoder/conv2d_3/Conv2D/ReadVariableOp?'encoder/conv2d_4/BiasAdd/ReadVariableOp?&encoder/conv2d_4/Conv2D/ReadVariableOp?'encoder/conv2d_5/BiasAdd/ReadVariableOp?&encoder/conv2d_5/Conv2D/ReadVariableOp?,encoder/latent_vector/BiasAdd/ReadVariableOp?+encoder/latent_vector/MatMul/ReadVariableOp?
&encoder/conv2d_3/Conv2D/ReadVariableOpReadVariableOp/encoder_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02(
&encoder/conv2d_3/Conv2D/ReadVariableOp?
encoder/conv2d_3/Conv2DConv2Dinputs.encoder/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
encoder/conv2d_3/Conv2D?
'encoder/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp0encoder_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02)
'encoder/conv2d_3/BiasAdd/ReadVariableOp?
encoder/conv2d_3/BiasAddBiasAdd encoder/conv2d_3/Conv2D:output:0/encoder/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
encoder/conv2d_3/BiasAdd?
encoder/conv2d_3/ReluRelu!encoder/conv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
encoder/conv2d_3/Relu?
&encoder/conv2d_4/Conv2D/ReadVariableOpReadVariableOp/encoder_conv2d_4_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02(
&encoder/conv2d_4/Conv2D/ReadVariableOp?
encoder/conv2d_4/Conv2DConv2D#encoder/conv2d_3/Relu:activations:0.encoder/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
encoder/conv2d_4/Conv2D?
'encoder/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp0encoder_conv2d_4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02)
'encoder/conv2d_4/BiasAdd/ReadVariableOp?
encoder/conv2d_4/BiasAddBiasAdd encoder/conv2d_4/Conv2D:output:0/encoder/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
encoder/conv2d_4/BiasAdd?
encoder/conv2d_4/ReluRelu!encoder/conv2d_4/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
encoder/conv2d_4/Relu?
&encoder/conv2d_5/Conv2D/ReadVariableOpReadVariableOp/encoder_conv2d_5_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02(
&encoder/conv2d_5/Conv2D/ReadVariableOp?
encoder/conv2d_5/Conv2DConv2D#encoder/conv2d_4/Relu:activations:0.encoder/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
encoder/conv2d_5/Conv2D?
'encoder/conv2d_5/BiasAdd/ReadVariableOpReadVariableOp0encoder_conv2d_5_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02)
'encoder/conv2d_5/BiasAdd/ReadVariableOp?
encoder/conv2d_5/BiasAddBiasAdd encoder/conv2d_5/Conv2D:output:0/encoder/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
encoder/conv2d_5/BiasAdd?
encoder/conv2d_5/ReluRelu!encoder/conv2d_5/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
encoder/conv2d_5/Relu?
encoder/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
encoder/flatten_1/Const?
encoder/flatten_1/ReshapeReshape#encoder/conv2d_5/Relu:activations:0 encoder/flatten_1/Const:output:0*
T0*(
_output_shapes
:?????????? 2
encoder/flatten_1/Reshape?
+encoder/latent_vector/MatMul/ReadVariableOpReadVariableOp4encoder_latent_vector_matmul_readvariableop_resource* 
_output_shapes
:
? ?*
dtype02-
+encoder/latent_vector/MatMul/ReadVariableOp?
encoder/latent_vector/MatMulMatMul"encoder/flatten_1/Reshape:output:03encoder/latent_vector/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
encoder/latent_vector/MatMul?
,encoder/latent_vector/BiasAdd/ReadVariableOpReadVariableOp5encoder_latent_vector_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02.
,encoder/latent_vector/BiasAdd/ReadVariableOp?
encoder/latent_vector/BiasAddBiasAdd&encoder/latent_vector/MatMul:product:04encoder/latent_vector/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
encoder/latent_vector/BiasAdd?
%decoder/dense_1/MatMul/ReadVariableOpReadVariableOp.decoder_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
?? *
dtype02'
%decoder/dense_1/MatMul/ReadVariableOp?
decoder/dense_1/MatMulMatMul&encoder/latent_vector/BiasAdd:output:0-decoder/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????? 2
decoder/dense_1/MatMul?
&decoder/dense_1/BiasAdd/ReadVariableOpReadVariableOp/decoder_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:? *
dtype02(
&decoder/dense_1/BiasAdd/ReadVariableOp?
decoder/dense_1/BiasAddBiasAdd decoder/dense_1/MatMul:product:0.decoder/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????? 2
decoder/dense_1/BiasAdd?
decoder/reshape_1/ShapeShape decoder/dense_1/BiasAdd:output:0*
T0*
_output_shapes
:2
decoder/reshape_1/Shape?
%decoder/reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%decoder/reshape_1/strided_slice/stack?
'decoder/reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'decoder/reshape_1/strided_slice/stack_1?
'decoder/reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'decoder/reshape_1/strided_slice/stack_2?
decoder/reshape_1/strided_sliceStridedSlice decoder/reshape_1/Shape:output:0.decoder/reshape_1/strided_slice/stack:output:00decoder/reshape_1/strided_slice/stack_1:output:00decoder/reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
decoder/reshape_1/strided_slice?
!decoder/reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2#
!decoder/reshape_1/Reshape/shape/1?
!decoder/reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2#
!decoder/reshape_1/Reshape/shape/2?
!decoder/reshape_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :?2#
!decoder/reshape_1/Reshape/shape/3?
decoder/reshape_1/Reshape/shapePack(decoder/reshape_1/strided_slice:output:0*decoder/reshape_1/Reshape/shape/1:output:0*decoder/reshape_1/Reshape/shape/2:output:0*decoder/reshape_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2!
decoder/reshape_1/Reshape/shape?
decoder/reshape_1/ReshapeReshape decoder/dense_1/BiasAdd:output:0(decoder/reshape_1/Reshape/shape:output:0*
T0*0
_output_shapes
:??????????2
decoder/reshape_1/Reshape?
 decoder/conv2d_transpose_3/ShapeShape"decoder/reshape_1/Reshape:output:0*
T0*
_output_shapes
:2"
 decoder/conv2d_transpose_3/Shape?
.decoder/conv2d_transpose_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 20
.decoder/conv2d_transpose_3/strided_slice/stack?
0decoder/conv2d_transpose_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0decoder/conv2d_transpose_3/strided_slice/stack_1?
0decoder/conv2d_transpose_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0decoder/conv2d_transpose_3/strided_slice/stack_2?
(decoder/conv2d_transpose_3/strided_sliceStridedSlice)decoder/conv2d_transpose_3/Shape:output:07decoder/conv2d_transpose_3/strided_slice/stack:output:09decoder/conv2d_transpose_3/strided_slice/stack_1:output:09decoder/conv2d_transpose_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(decoder/conv2d_transpose_3/strided_slice?
0decoder/conv2d_transpose_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:22
0decoder/conv2d_transpose_3/strided_slice_1/stack?
2decoder/conv2d_transpose_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2decoder/conv2d_transpose_3/strided_slice_1/stack_1?
2decoder/conv2d_transpose_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2decoder/conv2d_transpose_3/strided_slice_1/stack_2?
*decoder/conv2d_transpose_3/strided_slice_1StridedSlice)decoder/conv2d_transpose_3/Shape:output:09decoder/conv2d_transpose_3/strided_slice_1/stack:output:0;decoder/conv2d_transpose_3/strided_slice_1/stack_1:output:0;decoder/conv2d_transpose_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*decoder/conv2d_transpose_3/strided_slice_1?
0decoder/conv2d_transpose_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:22
0decoder/conv2d_transpose_3/strided_slice_2/stack?
2decoder/conv2d_transpose_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2decoder/conv2d_transpose_3/strided_slice_2/stack_1?
2decoder/conv2d_transpose_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2decoder/conv2d_transpose_3/strided_slice_2/stack_2?
*decoder/conv2d_transpose_3/strided_slice_2StridedSlice)decoder/conv2d_transpose_3/Shape:output:09decoder/conv2d_transpose_3/strided_slice_2/stack:output:0;decoder/conv2d_transpose_3/strided_slice_2/stack_1:output:0;decoder/conv2d_transpose_3/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*decoder/conv2d_transpose_3/strided_slice_2?
 decoder/conv2d_transpose_3/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2"
 decoder/conv2d_transpose_3/mul/y?
decoder/conv2d_transpose_3/mulMul3decoder/conv2d_transpose_3/strided_slice_1:output:0)decoder/conv2d_transpose_3/mul/y:output:0*
T0*
_output_shapes
: 2 
decoder/conv2d_transpose_3/mul?
"decoder/conv2d_transpose_3/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2$
"decoder/conv2d_transpose_3/mul_1/y?
 decoder/conv2d_transpose_3/mul_1Mul3decoder/conv2d_transpose_3/strided_slice_2:output:0+decoder/conv2d_transpose_3/mul_1/y:output:0*
T0*
_output_shapes
: 2"
 decoder/conv2d_transpose_3/mul_1?
"decoder/conv2d_transpose_3/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2$
"decoder/conv2d_transpose_3/stack/3?
 decoder/conv2d_transpose_3/stackPack1decoder/conv2d_transpose_3/strided_slice:output:0"decoder/conv2d_transpose_3/mul:z:0$decoder/conv2d_transpose_3/mul_1:z:0+decoder/conv2d_transpose_3/stack/3:output:0*
N*
T0*
_output_shapes
:2"
 decoder/conv2d_transpose_3/stack?
0decoder/conv2d_transpose_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0decoder/conv2d_transpose_3/strided_slice_3/stack?
2decoder/conv2d_transpose_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2decoder/conv2d_transpose_3/strided_slice_3/stack_1?
2decoder/conv2d_transpose_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2decoder/conv2d_transpose_3/strided_slice_3/stack_2?
*decoder/conv2d_transpose_3/strided_slice_3StridedSlice)decoder/conv2d_transpose_3/stack:output:09decoder/conv2d_transpose_3/strided_slice_3/stack:output:0;decoder/conv2d_transpose_3/strided_slice_3/stack_1:output:0;decoder/conv2d_transpose_3/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*decoder/conv2d_transpose_3/strided_slice_3?
:decoder/conv2d_transpose_3/conv2d_transpose/ReadVariableOpReadVariableOpCdecoder_conv2d_transpose_3_conv2d_transpose_readvariableop_resource*'
_output_shapes
:?*
dtype02<
:decoder/conv2d_transpose_3/conv2d_transpose/ReadVariableOp?
+decoder/conv2d_transpose_3/conv2d_transposeConv2DBackpropInput)decoder/conv2d_transpose_3/stack:output:0Bdecoder/conv2d_transpose_3/conv2d_transpose/ReadVariableOp:value:0"decoder/reshape_1/Reshape:output:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2-
+decoder/conv2d_transpose_3/conv2d_transpose?
1decoder/conv2d_transpose_3/BiasAdd/ReadVariableOpReadVariableOp:decoder_conv2d_transpose_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype023
1decoder/conv2d_transpose_3/BiasAdd/ReadVariableOp?
"decoder/conv2d_transpose_3/BiasAddBiasAdd4decoder/conv2d_transpose_3/conv2d_transpose:output:09decoder/conv2d_transpose_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2$
"decoder/conv2d_transpose_3/BiasAdd?
decoder/conv2d_transpose_3/ReluRelu+decoder/conv2d_transpose_3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2!
decoder/conv2d_transpose_3/Relu?
 decoder/conv2d_transpose_4/ShapeShape-decoder/conv2d_transpose_3/Relu:activations:0*
T0*
_output_shapes
:2"
 decoder/conv2d_transpose_4/Shape?
.decoder/conv2d_transpose_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 20
.decoder/conv2d_transpose_4/strided_slice/stack?
0decoder/conv2d_transpose_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0decoder/conv2d_transpose_4/strided_slice/stack_1?
0decoder/conv2d_transpose_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0decoder/conv2d_transpose_4/strided_slice/stack_2?
(decoder/conv2d_transpose_4/strided_sliceStridedSlice)decoder/conv2d_transpose_4/Shape:output:07decoder/conv2d_transpose_4/strided_slice/stack:output:09decoder/conv2d_transpose_4/strided_slice/stack_1:output:09decoder/conv2d_transpose_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(decoder/conv2d_transpose_4/strided_slice?
0decoder/conv2d_transpose_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:22
0decoder/conv2d_transpose_4/strided_slice_1/stack?
2decoder/conv2d_transpose_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2decoder/conv2d_transpose_4/strided_slice_1/stack_1?
2decoder/conv2d_transpose_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2decoder/conv2d_transpose_4/strided_slice_1/stack_2?
*decoder/conv2d_transpose_4/strided_slice_1StridedSlice)decoder/conv2d_transpose_4/Shape:output:09decoder/conv2d_transpose_4/strided_slice_1/stack:output:0;decoder/conv2d_transpose_4/strided_slice_1/stack_1:output:0;decoder/conv2d_transpose_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*decoder/conv2d_transpose_4/strided_slice_1?
0decoder/conv2d_transpose_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:22
0decoder/conv2d_transpose_4/strided_slice_2/stack?
2decoder/conv2d_transpose_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2decoder/conv2d_transpose_4/strided_slice_2/stack_1?
2decoder/conv2d_transpose_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2decoder/conv2d_transpose_4/strided_slice_2/stack_2?
*decoder/conv2d_transpose_4/strided_slice_2StridedSlice)decoder/conv2d_transpose_4/Shape:output:09decoder/conv2d_transpose_4/strided_slice_2/stack:output:0;decoder/conv2d_transpose_4/strided_slice_2/stack_1:output:0;decoder/conv2d_transpose_4/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*decoder/conv2d_transpose_4/strided_slice_2?
 decoder/conv2d_transpose_4/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2"
 decoder/conv2d_transpose_4/mul/y?
decoder/conv2d_transpose_4/mulMul3decoder/conv2d_transpose_4/strided_slice_1:output:0)decoder/conv2d_transpose_4/mul/y:output:0*
T0*
_output_shapes
: 2 
decoder/conv2d_transpose_4/mul?
"decoder/conv2d_transpose_4/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2$
"decoder/conv2d_transpose_4/mul_1/y?
 decoder/conv2d_transpose_4/mul_1Mul3decoder/conv2d_transpose_4/strided_slice_2:output:0+decoder/conv2d_transpose_4/mul_1/y:output:0*
T0*
_output_shapes
: 2"
 decoder/conv2d_transpose_4/mul_1?
"decoder/conv2d_transpose_4/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2$
"decoder/conv2d_transpose_4/stack/3?
 decoder/conv2d_transpose_4/stackPack1decoder/conv2d_transpose_4/strided_slice:output:0"decoder/conv2d_transpose_4/mul:z:0$decoder/conv2d_transpose_4/mul_1:z:0+decoder/conv2d_transpose_4/stack/3:output:0*
N*
T0*
_output_shapes
:2"
 decoder/conv2d_transpose_4/stack?
0decoder/conv2d_transpose_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0decoder/conv2d_transpose_4/strided_slice_3/stack?
2decoder/conv2d_transpose_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2decoder/conv2d_transpose_4/strided_slice_3/stack_1?
2decoder/conv2d_transpose_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2decoder/conv2d_transpose_4/strided_slice_3/stack_2?
*decoder/conv2d_transpose_4/strided_slice_3StridedSlice)decoder/conv2d_transpose_4/stack:output:09decoder/conv2d_transpose_4/strided_slice_3/stack:output:0;decoder/conv2d_transpose_4/strided_slice_3/stack_1:output:0;decoder/conv2d_transpose_4/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*decoder/conv2d_transpose_4/strided_slice_3?
:decoder/conv2d_transpose_4/conv2d_transpose/ReadVariableOpReadVariableOpCdecoder_conv2d_transpose_4_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02<
:decoder/conv2d_transpose_4/conv2d_transpose/ReadVariableOp?
+decoder/conv2d_transpose_4/conv2d_transposeConv2DBackpropInput)decoder/conv2d_transpose_4/stack:output:0Bdecoder/conv2d_transpose_4/conv2d_transpose/ReadVariableOp:value:0-decoder/conv2d_transpose_3/Relu:activations:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2-
+decoder/conv2d_transpose_4/conv2d_transpose?
1decoder/conv2d_transpose_4/BiasAdd/ReadVariableOpReadVariableOp:decoder_conv2d_transpose_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype023
1decoder/conv2d_transpose_4/BiasAdd/ReadVariableOp?
"decoder/conv2d_transpose_4/BiasAddBiasAdd4decoder/conv2d_transpose_4/conv2d_transpose:output:09decoder/conv2d_transpose_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2$
"decoder/conv2d_transpose_4/BiasAdd?
decoder/conv2d_transpose_4/ReluRelu+decoder/conv2d_transpose_4/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2!
decoder/conv2d_transpose_4/Relu?
 decoder/conv2d_transpose_5/ShapeShape-decoder/conv2d_transpose_4/Relu:activations:0*
T0*
_output_shapes
:2"
 decoder/conv2d_transpose_5/Shape?
.decoder/conv2d_transpose_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 20
.decoder/conv2d_transpose_5/strided_slice/stack?
0decoder/conv2d_transpose_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0decoder/conv2d_transpose_5/strided_slice/stack_1?
0decoder/conv2d_transpose_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0decoder/conv2d_transpose_5/strided_slice/stack_2?
(decoder/conv2d_transpose_5/strided_sliceStridedSlice)decoder/conv2d_transpose_5/Shape:output:07decoder/conv2d_transpose_5/strided_slice/stack:output:09decoder/conv2d_transpose_5/strided_slice/stack_1:output:09decoder/conv2d_transpose_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(decoder/conv2d_transpose_5/strided_slice?
0decoder/conv2d_transpose_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:22
0decoder/conv2d_transpose_5/strided_slice_1/stack?
2decoder/conv2d_transpose_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2decoder/conv2d_transpose_5/strided_slice_1/stack_1?
2decoder/conv2d_transpose_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2decoder/conv2d_transpose_5/strided_slice_1/stack_2?
*decoder/conv2d_transpose_5/strided_slice_1StridedSlice)decoder/conv2d_transpose_5/Shape:output:09decoder/conv2d_transpose_5/strided_slice_1/stack:output:0;decoder/conv2d_transpose_5/strided_slice_1/stack_1:output:0;decoder/conv2d_transpose_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*decoder/conv2d_transpose_5/strided_slice_1?
0decoder/conv2d_transpose_5/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:22
0decoder/conv2d_transpose_5/strided_slice_2/stack?
2decoder/conv2d_transpose_5/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2decoder/conv2d_transpose_5/strided_slice_2/stack_1?
2decoder/conv2d_transpose_5/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2decoder/conv2d_transpose_5/strided_slice_2/stack_2?
*decoder/conv2d_transpose_5/strided_slice_2StridedSlice)decoder/conv2d_transpose_5/Shape:output:09decoder/conv2d_transpose_5/strided_slice_2/stack:output:0;decoder/conv2d_transpose_5/strided_slice_2/stack_1:output:0;decoder/conv2d_transpose_5/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*decoder/conv2d_transpose_5/strided_slice_2?
 decoder/conv2d_transpose_5/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2"
 decoder/conv2d_transpose_5/mul/y?
decoder/conv2d_transpose_5/mulMul3decoder/conv2d_transpose_5/strided_slice_1:output:0)decoder/conv2d_transpose_5/mul/y:output:0*
T0*
_output_shapes
: 2 
decoder/conv2d_transpose_5/mul?
"decoder/conv2d_transpose_5/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2$
"decoder/conv2d_transpose_5/mul_1/y?
 decoder/conv2d_transpose_5/mul_1Mul3decoder/conv2d_transpose_5/strided_slice_2:output:0+decoder/conv2d_transpose_5/mul_1/y:output:0*
T0*
_output_shapes
: 2"
 decoder/conv2d_transpose_5/mul_1?
"decoder/conv2d_transpose_5/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2$
"decoder/conv2d_transpose_5/stack/3?
 decoder/conv2d_transpose_5/stackPack1decoder/conv2d_transpose_5/strided_slice:output:0"decoder/conv2d_transpose_5/mul:z:0$decoder/conv2d_transpose_5/mul_1:z:0+decoder/conv2d_transpose_5/stack/3:output:0*
N*
T0*
_output_shapes
:2"
 decoder/conv2d_transpose_5/stack?
0decoder/conv2d_transpose_5/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0decoder/conv2d_transpose_5/strided_slice_3/stack?
2decoder/conv2d_transpose_5/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2decoder/conv2d_transpose_5/strided_slice_3/stack_1?
2decoder/conv2d_transpose_5/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2decoder/conv2d_transpose_5/strided_slice_3/stack_2?
*decoder/conv2d_transpose_5/strided_slice_3StridedSlice)decoder/conv2d_transpose_5/stack:output:09decoder/conv2d_transpose_5/strided_slice_3/stack:output:0;decoder/conv2d_transpose_5/strided_slice_3/stack_1:output:0;decoder/conv2d_transpose_5/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*decoder/conv2d_transpose_5/strided_slice_3?
:decoder/conv2d_transpose_5/conv2d_transpose/ReadVariableOpReadVariableOpCdecoder_conv2d_transpose_5_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02<
:decoder/conv2d_transpose_5/conv2d_transpose/ReadVariableOp?
+decoder/conv2d_transpose_5/conv2d_transposeConv2DBackpropInput)decoder/conv2d_transpose_5/stack:output:0Bdecoder/conv2d_transpose_5/conv2d_transpose/ReadVariableOp:value:0-decoder/conv2d_transpose_4/Relu:activations:0*
T0*/
_output_shapes
:?????????  *
paddingSAME*
strides
2-
+decoder/conv2d_transpose_5/conv2d_transpose?
1decoder/conv2d_transpose_5/BiasAdd/ReadVariableOpReadVariableOp:decoder_conv2d_transpose_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype023
1decoder/conv2d_transpose_5/BiasAdd/ReadVariableOp?
"decoder/conv2d_transpose_5/BiasAddBiasAdd4decoder/conv2d_transpose_5/conv2d_transpose:output:09decoder/conv2d_transpose_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  2$
"decoder/conv2d_transpose_5/BiasAdd?
decoder/conv2d_transpose_5/ReluRelu+decoder/conv2d_transpose_5/BiasAdd:output:0*
T0*/
_output_shapes
:?????????  2!
decoder/conv2d_transpose_5/Relu?
decoder/decoder_output/ShapeShape-decoder/conv2d_transpose_5/Relu:activations:0*
T0*
_output_shapes
:2
decoder/decoder_output/Shape?
*decoder/decoder_output/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*decoder/decoder_output/strided_slice/stack?
,decoder/decoder_output/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,decoder/decoder_output/strided_slice/stack_1?
,decoder/decoder_output/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,decoder/decoder_output/strided_slice/stack_2?
$decoder/decoder_output/strided_sliceStridedSlice%decoder/decoder_output/Shape:output:03decoder/decoder_output/strided_slice/stack:output:05decoder/decoder_output/strided_slice/stack_1:output:05decoder/decoder_output/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$decoder/decoder_output/strided_slice?
,decoder/decoder_output/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2.
,decoder/decoder_output/strided_slice_1/stack?
.decoder/decoder_output/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.decoder/decoder_output/strided_slice_1/stack_1?
.decoder/decoder_output/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.decoder/decoder_output/strided_slice_1/stack_2?
&decoder/decoder_output/strided_slice_1StridedSlice%decoder/decoder_output/Shape:output:05decoder/decoder_output/strided_slice_1/stack:output:07decoder/decoder_output/strided_slice_1/stack_1:output:07decoder/decoder_output/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&decoder/decoder_output/strided_slice_1?
,decoder/decoder_output/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2.
,decoder/decoder_output/strided_slice_2/stack?
.decoder/decoder_output/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.decoder/decoder_output/strided_slice_2/stack_1?
.decoder/decoder_output/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.decoder/decoder_output/strided_slice_2/stack_2?
&decoder/decoder_output/strided_slice_2StridedSlice%decoder/decoder_output/Shape:output:05decoder/decoder_output/strided_slice_2/stack:output:07decoder/decoder_output/strided_slice_2/stack_1:output:07decoder/decoder_output/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&decoder/decoder_output/strided_slice_2~
decoder/decoder_output/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
decoder/decoder_output/mul/y?
decoder/decoder_output/mulMul/decoder/decoder_output/strided_slice_1:output:0%decoder/decoder_output/mul/y:output:0*
T0*
_output_shapes
: 2
decoder/decoder_output/mul?
decoder/decoder_output/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2 
decoder/decoder_output/mul_1/y?
decoder/decoder_output/mul_1Mul/decoder/decoder_output/strided_slice_2:output:0'decoder/decoder_output/mul_1/y:output:0*
T0*
_output_shapes
: 2
decoder/decoder_output/mul_1?
decoder/decoder_output/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2 
decoder/decoder_output/stack/3?
decoder/decoder_output/stackPack-decoder/decoder_output/strided_slice:output:0decoder/decoder_output/mul:z:0 decoder/decoder_output/mul_1:z:0'decoder/decoder_output/stack/3:output:0*
N*
T0*
_output_shapes
:2
decoder/decoder_output/stack?
,decoder/decoder_output/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,decoder/decoder_output/strided_slice_3/stack?
.decoder/decoder_output/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.decoder/decoder_output/strided_slice_3/stack_1?
.decoder/decoder_output/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.decoder/decoder_output/strided_slice_3/stack_2?
&decoder/decoder_output/strided_slice_3StridedSlice%decoder/decoder_output/stack:output:05decoder/decoder_output/strided_slice_3/stack:output:07decoder/decoder_output/strided_slice_3/stack_1:output:07decoder/decoder_output/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&decoder/decoder_output/strided_slice_3?
6decoder/decoder_output/conv2d_transpose/ReadVariableOpReadVariableOp?decoder_decoder_output_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype028
6decoder/decoder_output/conv2d_transpose/ReadVariableOp?
'decoder/decoder_output/conv2d_transposeConv2DBackpropInput%decoder/decoder_output/stack:output:0>decoder/decoder_output/conv2d_transpose/ReadVariableOp:value:0-decoder/conv2d_transpose_5/Relu:activations:0*
T0*/
_output_shapes
:?????????  *
paddingSAME*
strides
2)
'decoder/decoder_output/conv2d_transpose?
-decoder/decoder_output/BiasAdd/ReadVariableOpReadVariableOp6decoder_decoder_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-decoder/decoder_output/BiasAdd/ReadVariableOp?
decoder/decoder_output/BiasAddBiasAdd0decoder/decoder_output/conv2d_transpose:output:05decoder/decoder_output/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  2 
decoder/decoder_output/BiasAdd?
decoder/decoder_output/SigmoidSigmoid'decoder/decoder_output/BiasAdd:output:0*
T0*/
_output_shapes
:?????????  2 
decoder/decoder_output/Sigmoid?
IdentityIdentity"decoder/decoder_output/Sigmoid:y:02^decoder/conv2d_transpose_3/BiasAdd/ReadVariableOp;^decoder/conv2d_transpose_3/conv2d_transpose/ReadVariableOp2^decoder/conv2d_transpose_4/BiasAdd/ReadVariableOp;^decoder/conv2d_transpose_4/conv2d_transpose/ReadVariableOp2^decoder/conv2d_transpose_5/BiasAdd/ReadVariableOp;^decoder/conv2d_transpose_5/conv2d_transpose/ReadVariableOp.^decoder/decoder_output/BiasAdd/ReadVariableOp7^decoder/decoder_output/conv2d_transpose/ReadVariableOp'^decoder/dense_1/BiasAdd/ReadVariableOp&^decoder/dense_1/MatMul/ReadVariableOp(^encoder/conv2d_3/BiasAdd/ReadVariableOp'^encoder/conv2d_3/Conv2D/ReadVariableOp(^encoder/conv2d_4/BiasAdd/ReadVariableOp'^encoder/conv2d_4/Conv2D/ReadVariableOp(^encoder/conv2d_5/BiasAdd/ReadVariableOp'^encoder/conv2d_5/Conv2D/ReadVariableOp-^encoder/latent_vector/BiasAdd/ReadVariableOp,^encoder/latent_vector/MatMul/ReadVariableOp*
T0*/
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*v
_input_shapese
c:?????????  ::::::::::::::::::2f
1decoder/conv2d_transpose_3/BiasAdd/ReadVariableOp1decoder/conv2d_transpose_3/BiasAdd/ReadVariableOp2x
:decoder/conv2d_transpose_3/conv2d_transpose/ReadVariableOp:decoder/conv2d_transpose_3/conv2d_transpose/ReadVariableOp2f
1decoder/conv2d_transpose_4/BiasAdd/ReadVariableOp1decoder/conv2d_transpose_4/BiasAdd/ReadVariableOp2x
:decoder/conv2d_transpose_4/conv2d_transpose/ReadVariableOp:decoder/conv2d_transpose_4/conv2d_transpose/ReadVariableOp2f
1decoder/conv2d_transpose_5/BiasAdd/ReadVariableOp1decoder/conv2d_transpose_5/BiasAdd/ReadVariableOp2x
:decoder/conv2d_transpose_5/conv2d_transpose/ReadVariableOp:decoder/conv2d_transpose_5/conv2d_transpose/ReadVariableOp2^
-decoder/decoder_output/BiasAdd/ReadVariableOp-decoder/decoder_output/BiasAdd/ReadVariableOp2p
6decoder/decoder_output/conv2d_transpose/ReadVariableOp6decoder/decoder_output/conv2d_transpose/ReadVariableOp2P
&decoder/dense_1/BiasAdd/ReadVariableOp&decoder/dense_1/BiasAdd/ReadVariableOp2N
%decoder/dense_1/MatMul/ReadVariableOp%decoder/dense_1/MatMul/ReadVariableOp2R
'encoder/conv2d_3/BiasAdd/ReadVariableOp'encoder/conv2d_3/BiasAdd/ReadVariableOp2P
&encoder/conv2d_3/Conv2D/ReadVariableOp&encoder/conv2d_3/Conv2D/ReadVariableOp2R
'encoder/conv2d_4/BiasAdd/ReadVariableOp'encoder/conv2d_4/BiasAdd/ReadVariableOp2P
&encoder/conv2d_4/Conv2D/ReadVariableOp&encoder/conv2d_4/Conv2D/ReadVariableOp2R
'encoder/conv2d_5/BiasAdd/ReadVariableOp'encoder/conv2d_5/BiasAdd/ReadVariableOp2P
&encoder/conv2d_5/Conv2D/ReadVariableOp&encoder/conv2d_5/Conv2D/ReadVariableOp2\
,encoder/latent_vector/BiasAdd/ReadVariableOp,encoder/latent_vector/BiasAdd/ReadVariableOp2Z
+encoder/latent_vector/MatMul/ReadVariableOp+encoder/latent_vector/MatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
??
?!
!__inference__traced_restore_89694
file_prefix
assignvariableop_adam_iter"
assignvariableop_1_adam_beta_1"
assignvariableop_2_adam_beta_2!
assignvariableop_3_adam_decay)
%assignvariableop_4_adam_learning_rate&
"assignvariableop_5_conv2d_3_kernel$
 assignvariableop_6_conv2d_3_bias&
"assignvariableop_7_conv2d_4_kernel$
 assignvariableop_8_conv2d_4_bias&
"assignvariableop_9_conv2d_5_kernel%
!assignvariableop_10_conv2d_5_bias.
*assignvariableop_11_latent_vector_1_kernel,
(assignvariableop_12_latent_vector_1_bias&
"assignvariableop_13_dense_1_kernel$
 assignvariableop_14_dense_1_bias1
-assignvariableop_15_conv2d_transpose_3_kernel/
+assignvariableop_16_conv2d_transpose_3_bias1
-assignvariableop_17_conv2d_transpose_4_kernel/
+assignvariableop_18_conv2d_transpose_4_bias1
-assignvariableop_19_conv2d_transpose_5_kernel/
+assignvariableop_20_conv2d_transpose_5_bias/
+assignvariableop_21_decoder_output_1_kernel-
)assignvariableop_22_decoder_output_1_bias.
*assignvariableop_23_adam_conv2d_3_kernel_m,
(assignvariableop_24_adam_conv2d_3_bias_m.
*assignvariableop_25_adam_conv2d_4_kernel_m,
(assignvariableop_26_adam_conv2d_4_bias_m.
*assignvariableop_27_adam_conv2d_5_kernel_m,
(assignvariableop_28_adam_conv2d_5_bias_m5
1assignvariableop_29_adam_latent_vector_1_kernel_m3
/assignvariableop_30_adam_latent_vector_1_bias_m-
)assignvariableop_31_adam_dense_1_kernel_m+
'assignvariableop_32_adam_dense_1_bias_m8
4assignvariableop_33_adam_conv2d_transpose_3_kernel_m6
2assignvariableop_34_adam_conv2d_transpose_3_bias_m8
4assignvariableop_35_adam_conv2d_transpose_4_kernel_m6
2assignvariableop_36_adam_conv2d_transpose_4_bias_m8
4assignvariableop_37_adam_conv2d_transpose_5_kernel_m6
2assignvariableop_38_adam_conv2d_transpose_5_bias_m6
2assignvariableop_39_adam_decoder_output_1_kernel_m4
0assignvariableop_40_adam_decoder_output_1_bias_m.
*assignvariableop_41_adam_conv2d_3_kernel_v,
(assignvariableop_42_adam_conv2d_3_bias_v.
*assignvariableop_43_adam_conv2d_4_kernel_v,
(assignvariableop_44_adam_conv2d_4_bias_v.
*assignvariableop_45_adam_conv2d_5_kernel_v,
(assignvariableop_46_adam_conv2d_5_bias_v5
1assignvariableop_47_adam_latent_vector_1_kernel_v3
/assignvariableop_48_adam_latent_vector_1_bias_v-
)assignvariableop_49_adam_dense_1_kernel_v+
'assignvariableop_50_adam_dense_1_bias_v8
4assignvariableop_51_adam_conv2d_transpose_3_kernel_v6
2assignvariableop_52_adam_conv2d_transpose_3_bias_v8
4assignvariableop_53_adam_conv2d_transpose_4_kernel_v6
2assignvariableop_54_adam_conv2d_transpose_4_bias_v8
4assignvariableop_55_adam_conv2d_transpose_5_kernel_v6
2assignvariableop_56_adam_conv2d_transpose_5_bias_v6
2assignvariableop_57_adam_decoder_output_1_kernel_v4
0assignvariableop_58_adam_decoder_output_1_bias_v
identity_60??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_55?AssignVariableOp_56?AssignVariableOp_57?AssignVariableOp_58?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?	RestoreV2?RestoreV2_1?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:;*
dtype0*?
value?B?;B)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:;*
dtype0*?
value?B~;B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*I
dtypes?
=2;	2
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0	*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_adam_iterIdentity:output:0*
_output_shapes
 *
dtype0	2
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_adam_beta_1Identity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_beta_2Identity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_decayIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp%assignvariableop_4_adam_learning_rateIdentity_4:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp"assignvariableop_5_conv2d_3_kernelIdentity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5\

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp assignvariableop_6_conv2d_3_biasIdentity_6:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_6\

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp"assignvariableop_7_conv2d_4_kernelIdentity_7:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_7\

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp assignvariableop_8_conv2d_4_biasIdentity_8:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_8\

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp"assignvariableop_9_conv2d_5_kernelIdentity_9:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_9_
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp!assignvariableop_10_conv2d_5_biasIdentity_10:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_10_
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp*assignvariableop_11_latent_vector_1_kernelIdentity_11:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_11_
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp(assignvariableop_12_latent_vector_1_biasIdentity_12:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_12_
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp"assignvariableop_13_dense_1_kernelIdentity_13:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_13_
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp assignvariableop_14_dense_1_biasIdentity_14:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_14_
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp-assignvariableop_15_conv2d_transpose_3_kernelIdentity_15:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_15_
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp+assignvariableop_16_conv2d_transpose_3_biasIdentity_16:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_16_
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp-assignvariableop_17_conv2d_transpose_4_kernelIdentity_17:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_17_
Identity_18IdentityRestoreV2:tensors:18*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp+assignvariableop_18_conv2d_transpose_4_biasIdentity_18:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_18_
Identity_19IdentityRestoreV2:tensors:19*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp-assignvariableop_19_conv2d_transpose_5_kernelIdentity_19:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_19_
Identity_20IdentityRestoreV2:tensors:20*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp+assignvariableop_20_conv2d_transpose_5_biasIdentity_20:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_20_
Identity_21IdentityRestoreV2:tensors:21*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp+assignvariableop_21_decoder_output_1_kernelIdentity_21:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_21_
Identity_22IdentityRestoreV2:tensors:22*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp)assignvariableop_22_decoder_output_1_biasIdentity_22:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_22_
Identity_23IdentityRestoreV2:tensors:23*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp*assignvariableop_23_adam_conv2d_3_kernel_mIdentity_23:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_23_
Identity_24IdentityRestoreV2:tensors:24*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp(assignvariableop_24_adam_conv2d_3_bias_mIdentity_24:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_24_
Identity_25IdentityRestoreV2:tensors:25*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp*assignvariableop_25_adam_conv2d_4_kernel_mIdentity_25:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_25_
Identity_26IdentityRestoreV2:tensors:26*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp(assignvariableop_26_adam_conv2d_4_bias_mIdentity_26:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_26_
Identity_27IdentityRestoreV2:tensors:27*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp*assignvariableop_27_adam_conv2d_5_kernel_mIdentity_27:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_27_
Identity_28IdentityRestoreV2:tensors:28*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp(assignvariableop_28_adam_conv2d_5_bias_mIdentity_28:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_28_
Identity_29IdentityRestoreV2:tensors:29*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp1assignvariableop_29_adam_latent_vector_1_kernel_mIdentity_29:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_29_
Identity_30IdentityRestoreV2:tensors:30*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp/assignvariableop_30_adam_latent_vector_1_bias_mIdentity_30:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_30_
Identity_31IdentityRestoreV2:tensors:31*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp)assignvariableop_31_adam_dense_1_kernel_mIdentity_31:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_31_
Identity_32IdentityRestoreV2:tensors:32*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp'assignvariableop_32_adam_dense_1_bias_mIdentity_32:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_32_
Identity_33IdentityRestoreV2:tensors:33*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp4assignvariableop_33_adam_conv2d_transpose_3_kernel_mIdentity_33:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_33_
Identity_34IdentityRestoreV2:tensors:34*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp2assignvariableop_34_adam_conv2d_transpose_3_bias_mIdentity_34:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_34_
Identity_35IdentityRestoreV2:tensors:35*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp4assignvariableop_35_adam_conv2d_transpose_4_kernel_mIdentity_35:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_35_
Identity_36IdentityRestoreV2:tensors:36*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp2assignvariableop_36_adam_conv2d_transpose_4_bias_mIdentity_36:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_36_
Identity_37IdentityRestoreV2:tensors:37*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp4assignvariableop_37_adam_conv2d_transpose_5_kernel_mIdentity_37:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_37_
Identity_38IdentityRestoreV2:tensors:38*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp2assignvariableop_38_adam_conv2d_transpose_5_bias_mIdentity_38:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_38_
Identity_39IdentityRestoreV2:tensors:39*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp2assignvariableop_39_adam_decoder_output_1_kernel_mIdentity_39:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_39_
Identity_40IdentityRestoreV2:tensors:40*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp0assignvariableop_40_adam_decoder_output_1_bias_mIdentity_40:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_40_
Identity_41IdentityRestoreV2:tensors:41*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOp*assignvariableop_41_adam_conv2d_3_kernel_vIdentity_41:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_41_
Identity_42IdentityRestoreV2:tensors:42*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOp(assignvariableop_42_adam_conv2d_3_bias_vIdentity_42:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_42_
Identity_43IdentityRestoreV2:tensors:43*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOp*assignvariableop_43_adam_conv2d_4_kernel_vIdentity_43:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_43_
Identity_44IdentityRestoreV2:tensors:44*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOp(assignvariableop_44_adam_conv2d_4_bias_vIdentity_44:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_44_
Identity_45IdentityRestoreV2:tensors:45*
T0*
_output_shapes
:2
Identity_45?
AssignVariableOp_45AssignVariableOp*assignvariableop_45_adam_conv2d_5_kernel_vIdentity_45:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_45_
Identity_46IdentityRestoreV2:tensors:46*
T0*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOp(assignvariableop_46_adam_conv2d_5_bias_vIdentity_46:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_46_
Identity_47IdentityRestoreV2:tensors:47*
T0*
_output_shapes
:2
Identity_47?
AssignVariableOp_47AssignVariableOp1assignvariableop_47_adam_latent_vector_1_kernel_vIdentity_47:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_47_
Identity_48IdentityRestoreV2:tensors:48*
T0*
_output_shapes
:2
Identity_48?
AssignVariableOp_48AssignVariableOp/assignvariableop_48_adam_latent_vector_1_bias_vIdentity_48:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_48_
Identity_49IdentityRestoreV2:tensors:49*
T0*
_output_shapes
:2
Identity_49?
AssignVariableOp_49AssignVariableOp)assignvariableop_49_adam_dense_1_kernel_vIdentity_49:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_49_
Identity_50IdentityRestoreV2:tensors:50*
T0*
_output_shapes
:2
Identity_50?
AssignVariableOp_50AssignVariableOp'assignvariableop_50_adam_dense_1_bias_vIdentity_50:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_50_
Identity_51IdentityRestoreV2:tensors:51*
T0*
_output_shapes
:2
Identity_51?
AssignVariableOp_51AssignVariableOp4assignvariableop_51_adam_conv2d_transpose_3_kernel_vIdentity_51:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_51_
Identity_52IdentityRestoreV2:tensors:52*
T0*
_output_shapes
:2
Identity_52?
AssignVariableOp_52AssignVariableOp2assignvariableop_52_adam_conv2d_transpose_3_bias_vIdentity_52:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_52_
Identity_53IdentityRestoreV2:tensors:53*
T0*
_output_shapes
:2
Identity_53?
AssignVariableOp_53AssignVariableOp4assignvariableop_53_adam_conv2d_transpose_4_kernel_vIdentity_53:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_53_
Identity_54IdentityRestoreV2:tensors:54*
T0*
_output_shapes
:2
Identity_54?
AssignVariableOp_54AssignVariableOp2assignvariableop_54_adam_conv2d_transpose_4_bias_vIdentity_54:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_54_
Identity_55IdentityRestoreV2:tensors:55*
T0*
_output_shapes
:2
Identity_55?
AssignVariableOp_55AssignVariableOp4assignvariableop_55_adam_conv2d_transpose_5_kernel_vIdentity_55:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_55_
Identity_56IdentityRestoreV2:tensors:56*
T0*
_output_shapes
:2
Identity_56?
AssignVariableOp_56AssignVariableOp2assignvariableop_56_adam_conv2d_transpose_5_bias_vIdentity_56:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_56_
Identity_57IdentityRestoreV2:tensors:57*
T0*
_output_shapes
:2
Identity_57?
AssignVariableOp_57AssignVariableOp2assignvariableop_57_adam_decoder_output_1_kernel_vIdentity_57:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_57_
Identity_58IdentityRestoreV2:tensors:58*
T0*
_output_shapes
:2
Identity_58?
AssignVariableOp_58AssignVariableOp0assignvariableop_58_adam_decoder_output_1_bias_vIdentity_58:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_58?
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2_1/tensor_names?
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2_1/shape_and_slices?
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
22
RestoreV2_19
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?

Identity_59Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_59?

Identity_60IdentityIdentity_59:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2
Identity_60"#
identity_60Identity_60:output:0*?
_input_shapes?
?: :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV22
RestoreV2_1RestoreV2_1:+ '
%
_user_specified_namefile_prefix
?
?
B__inference_encoder_layer_call_and_return_conditional_losses_87837
encoder_input+
'conv2d_3_statefulpartitionedcall_args_1+
'conv2d_3_statefulpartitionedcall_args_2+
'conv2d_4_statefulpartitionedcall_args_1+
'conv2d_4_statefulpartitionedcall_args_2+
'conv2d_5_statefulpartitionedcall_args_1+
'conv2d_5_statefulpartitionedcall_args_20
,latent_vector_statefulpartitionedcall_args_10
,latent_vector_statefulpartitionedcall_args_2
identity?? conv2d_3/StatefulPartitionedCall? conv2d_4/StatefulPartitionedCall? conv2d_5/StatefulPartitionedCall?%latent_vector/StatefulPartitionedCall?
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCallencoder_input'conv2d_3_statefulpartitionedcall_args_1'conv2d_3_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:?????????@*-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_conv2d_3_layer_call_and_return_conditional_losses_877202"
 conv2d_3/StatefulPartitionedCall?
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0'conv2d_4_statefulpartitionedcall_args_1'conv2d_4_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:??????????*-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_conv2d_4_layer_call_and_return_conditional_losses_877412"
 conv2d_4/StatefulPartitionedCall?
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0'conv2d_5_statefulpartitionedcall_args_1'conv2d_5_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:??????????*-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_conv2d_5_layer_call_and_return_conditional_losses_877622"
 conv2d_5/StatefulPartitionedCall?
flatten_1/PartitionedCallPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:?????????? *-
config_proto

GPU

CPU2*0J 8*M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_877892
flatten_1/PartitionedCall?
%latent_vector/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0,latent_vector_statefulpartitionedcall_args_1,latent_vector_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????*-
config_proto

GPU

CPU2*0J 8*Q
fLRJ
H__inference_latent_vector_layer_call_and_return_conditional_losses_878072'
%latent_vector/StatefulPartitionedCall?
IdentityIdentity.latent_vector/StatefulPartitionedCall:output:0!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall&^latent_vector/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????  ::::::::2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2N
%latent_vector/StatefulPartitionedCall%latent_vector/StatefulPartitionedCall:- )
'
_user_specified_nameencoder_input
?
?
B__inference_dense_1_layer_call_and_return_conditional_losses_88084

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
?? *
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????? 2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:? *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????? 2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:?????????? 2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
?
?
F__inference_autoencoder_layer_call_and_return_conditional_losses_88327
encoder_input*
&encoder_statefulpartitionedcall_args_1*
&encoder_statefulpartitionedcall_args_2*
&encoder_statefulpartitionedcall_args_3*
&encoder_statefulpartitionedcall_args_4*
&encoder_statefulpartitionedcall_args_5*
&encoder_statefulpartitionedcall_args_6*
&encoder_statefulpartitionedcall_args_7*
&encoder_statefulpartitionedcall_args_8*
&decoder_statefulpartitionedcall_args_1*
&decoder_statefulpartitionedcall_args_2*
&decoder_statefulpartitionedcall_args_3*
&decoder_statefulpartitionedcall_args_4*
&decoder_statefulpartitionedcall_args_5*
&decoder_statefulpartitionedcall_args_6*
&decoder_statefulpartitionedcall_args_7*
&decoder_statefulpartitionedcall_args_8*
&decoder_statefulpartitionedcall_args_9+
'decoder_statefulpartitionedcall_args_10
identity??decoder/StatefulPartitionedCall?encoder/StatefulPartitionedCall?
encoder/StatefulPartitionedCallStatefulPartitionedCallencoder_input&encoder_statefulpartitionedcall_args_1&encoder_statefulpartitionedcall_args_2&encoder_statefulpartitionedcall_args_3&encoder_statefulpartitionedcall_args_4&encoder_statefulpartitionedcall_args_5&encoder_statefulpartitionedcall_args_6&encoder_statefulpartitionedcall_args_7&encoder_statefulpartitionedcall_args_8*
Tin
2	*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????*-
config_proto

GPU

CPU2*0J 8*K
fFRD
B__inference_encoder_layer_call_and_return_conditional_losses_878872!
encoder/StatefulPartitionedCall?
decoder/StatefulPartitionedCallStatefulPartitionedCall(encoder/StatefulPartitionedCall:output:0&decoder_statefulpartitionedcall_args_1&decoder_statefulpartitionedcall_args_2&decoder_statefulpartitionedcall_args_3&decoder_statefulpartitionedcall_args_4&decoder_statefulpartitionedcall_args_5&decoder_statefulpartitionedcall_args_6&decoder_statefulpartitionedcall_args_7&decoder_statefulpartitionedcall_args_8&decoder_statefulpartitionedcall_args_9'decoder_statefulpartitionedcall_args_10*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+???????????????????????????*-
config_proto

GPU

CPU2*0J 8*K
fFRD
B__inference_decoder_layer_call_and_return_conditional_losses_882092!
decoder/StatefulPartitionedCall?
IdentityIdentity(decoder/StatefulPartitionedCall:output:0 ^decoder/StatefulPartitionedCall ^encoder/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*v
_input_shapese
c:?????????  ::::::::::::::::::2B
decoder/StatefulPartitionedCalldecoder/StatefulPartitionedCall2B
encoder/StatefulPartitionedCallencoder/StatefulPartitionedCall:- )
'
_user_specified_nameencoder_input
?
?
'__inference_decoder_layer_call_fn_89240

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+???????????????????????????*-
config_proto

GPU

CPU2*0J 8*K
fFRD
B__inference_decoder_layer_call_and_return_conditional_losses_882092
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*O
_input_shapes>
<:??????????::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?%
?
B__inference_decoder_layer_call_and_return_conditional_losses_88131
decoder_input*
&dense_1_statefulpartitionedcall_args_1*
&dense_1_statefulpartitionedcall_args_25
1conv2d_transpose_3_statefulpartitionedcall_args_15
1conv2d_transpose_3_statefulpartitionedcall_args_25
1conv2d_transpose_4_statefulpartitionedcall_args_15
1conv2d_transpose_4_statefulpartitionedcall_args_25
1conv2d_transpose_5_statefulpartitionedcall_args_15
1conv2d_transpose_5_statefulpartitionedcall_args_21
-decoder_output_statefulpartitionedcall_args_11
-decoder_output_statefulpartitionedcall_args_2
identity??*conv2d_transpose_3/StatefulPartitionedCall?*conv2d_transpose_4/StatefulPartitionedCall?*conv2d_transpose_5/StatefulPartitionedCall?&decoder_output/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCalldecoder_input&dense_1_statefulpartitionedcall_args_1&dense_1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:?????????? *-
config_proto

GPU

CPU2*0J 8*K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_880842!
dense_1/StatefulPartitionedCall?
reshape_1/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:??????????*-
config_proto

GPU

CPU2*0J 8*M
fHRF
D__inference_reshape_1_layer_call_and_return_conditional_losses_881102
reshape_1/PartitionedCall?
*conv2d_transpose_3/StatefulPartitionedCallStatefulPartitionedCall"reshape_1/PartitionedCall:output:01conv2d_transpose_3_statefulpartitionedcall_args_11conv2d_transpose_3_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+???????????????????????????*-
config_proto

GPU

CPU2*0J 8*V
fQRO
M__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_879332,
*conv2d_transpose_3/StatefulPartitionedCall?
*conv2d_transpose_4/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_3/StatefulPartitionedCall:output:01conv2d_transpose_4_statefulpartitionedcall_args_11conv2d_transpose_4_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+???????????????????????????*-
config_proto

GPU

CPU2*0J 8*V
fQRO
M__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_879762,
*conv2d_transpose_4/StatefulPartitionedCall?
*conv2d_transpose_5/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_4/StatefulPartitionedCall:output:01conv2d_transpose_5_statefulpartitionedcall_args_11conv2d_transpose_5_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+???????????????????????????*-
config_proto

GPU

CPU2*0J 8*V
fQRO
M__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_880192,
*conv2d_transpose_5/StatefulPartitionedCall?
&decoder_output/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_5/StatefulPartitionedCall:output:0-decoder_output_statefulpartitionedcall_args_1-decoder_output_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+???????????????????????????*-
config_proto

GPU

CPU2*0J 8*R
fMRK
I__inference_decoder_output_layer_call_and_return_conditional_losses_880622(
&decoder_output/StatefulPartitionedCall?
IdentityIdentity/decoder_output/StatefulPartitionedCall:output:0+^conv2d_transpose_3/StatefulPartitionedCall+^conv2d_transpose_4/StatefulPartitionedCall+^conv2d_transpose_5/StatefulPartitionedCall'^decoder_output/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*O
_input_shapes>
<:??????????::::::::::2X
*conv2d_transpose_3/StatefulPartitionedCall*conv2d_transpose_3/StatefulPartitionedCall2X
*conv2d_transpose_4/StatefulPartitionedCall*conv2d_transpose_4/StatefulPartitionedCall2X
*conv2d_transpose_5/StatefulPartitionedCall*conv2d_transpose_5/StatefulPartitionedCall2P
&decoder_output/StatefulPartitionedCall&decoder_output/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:- )
'
_user_specified_namedecoder_input
?
`
D__inference_reshape_1_layer_call_and_return_conditional_losses_88110

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2e
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :?2
Reshape/shape/3?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapex
ReshapeReshapeinputsReshape/shape:output:0*
T0*0
_output_shapes
:??????????2	
Reshapem
IdentityIdentityReshape:output:0*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:?????????? :& "
 
_user_specified_nameinputs
?
?
B__inference_encoder_layer_call_and_return_conditional_losses_87887

inputs+
'conv2d_3_statefulpartitionedcall_args_1+
'conv2d_3_statefulpartitionedcall_args_2+
'conv2d_4_statefulpartitionedcall_args_1+
'conv2d_4_statefulpartitionedcall_args_2+
'conv2d_5_statefulpartitionedcall_args_1+
'conv2d_5_statefulpartitionedcall_args_20
,latent_vector_statefulpartitionedcall_args_10
,latent_vector_statefulpartitionedcall_args_2
identity?? conv2d_3/StatefulPartitionedCall? conv2d_4/StatefulPartitionedCall? conv2d_5/StatefulPartitionedCall?%latent_vector/StatefulPartitionedCall?
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCallinputs'conv2d_3_statefulpartitionedcall_args_1'conv2d_3_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:?????????@*-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_conv2d_3_layer_call_and_return_conditional_losses_877202"
 conv2d_3/StatefulPartitionedCall?
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0'conv2d_4_statefulpartitionedcall_args_1'conv2d_4_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:??????????*-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_conv2d_4_layer_call_and_return_conditional_losses_877412"
 conv2d_4/StatefulPartitionedCall?
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0'conv2d_5_statefulpartitionedcall_args_1'conv2d_5_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:??????????*-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_conv2d_5_layer_call_and_return_conditional_losses_877622"
 conv2d_5/StatefulPartitionedCall?
flatten_1/PartitionedCallPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:?????????? *-
config_proto

GPU

CPU2*0J 8*M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_877892
flatten_1/PartitionedCall?
%latent_vector/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0,latent_vector_statefulpartitionedcall_args_1,latent_vector_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????*-
config_proto

GPU

CPU2*0J 8*Q
fLRJ
H__inference_latent_vector_layer_call_and_return_conditional_losses_878072'
%latent_vector/StatefulPartitionedCall?
IdentityIdentity.latent_vector/StatefulPartitionedCall:output:0!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall&^latent_vector/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????  ::::::::2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2N
%latent_vector/StatefulPartitionedCall%latent_vector/StatefulPartitionedCall:& "
 
_user_specified_nameinputs
?$
?
I__inference_decoder_output_layer_call_and_return_conditional_losses_88062

inputs,
(conv2d_transpose_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/3?
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
2
conv2d_transpose?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????2	
BiasAdd{
SigmoidSigmoidBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:& "
 
_user_specified_nameinputs
?
?
+__inference_autoencoder_layer_call_fn_88422
encoder_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallencoder_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+???????????????????????????*-
config_proto

GPU

CPU2*0J 8*O
fJRH
F__inference_autoencoder_layer_call_and_return_conditional_losses_884012
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*v
_input_shapese
c:?????????  ::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:- )
'
_user_specified_nameencoder_input
?
?
+__inference_autoencoder_layer_call_fn_88815

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+???????????????????????????*-
config_proto

GPU

CPU2*0J 8*O
fJRH
F__inference_autoencoder_layer_call_and_return_conditional_losses_883542
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*v
_input_shapese
c:?????????  ::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?
?
+__inference_autoencoder_layer_call_fn_88838

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+???????????????????????????*-
config_proto

GPU

CPU2*0J 8*O
fJRH
F__inference_autoencoder_layer_call_and_return_conditional_losses_884012
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*v
_input_shapese
c:?????????  ::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?%
?
B__inference_decoder_layer_call_and_return_conditional_losses_88151
decoder_input*
&dense_1_statefulpartitionedcall_args_1*
&dense_1_statefulpartitionedcall_args_25
1conv2d_transpose_3_statefulpartitionedcall_args_15
1conv2d_transpose_3_statefulpartitionedcall_args_25
1conv2d_transpose_4_statefulpartitionedcall_args_15
1conv2d_transpose_4_statefulpartitionedcall_args_25
1conv2d_transpose_5_statefulpartitionedcall_args_15
1conv2d_transpose_5_statefulpartitionedcall_args_21
-decoder_output_statefulpartitionedcall_args_11
-decoder_output_statefulpartitionedcall_args_2
identity??*conv2d_transpose_3/StatefulPartitionedCall?*conv2d_transpose_4/StatefulPartitionedCall?*conv2d_transpose_5/StatefulPartitionedCall?&decoder_output/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCalldecoder_input&dense_1_statefulpartitionedcall_args_1&dense_1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:?????????? *-
config_proto

GPU

CPU2*0J 8*K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_880842!
dense_1/StatefulPartitionedCall?
reshape_1/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:??????????*-
config_proto

GPU

CPU2*0J 8*M
fHRF
D__inference_reshape_1_layer_call_and_return_conditional_losses_881102
reshape_1/PartitionedCall?
*conv2d_transpose_3/StatefulPartitionedCallStatefulPartitionedCall"reshape_1/PartitionedCall:output:01conv2d_transpose_3_statefulpartitionedcall_args_11conv2d_transpose_3_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+???????????????????????????*-
config_proto

GPU

CPU2*0J 8*V
fQRO
M__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_879332,
*conv2d_transpose_3/StatefulPartitionedCall?
*conv2d_transpose_4/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_3/StatefulPartitionedCall:output:01conv2d_transpose_4_statefulpartitionedcall_args_11conv2d_transpose_4_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+???????????????????????????*-
config_proto

GPU

CPU2*0J 8*V
fQRO
M__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_879762,
*conv2d_transpose_4/StatefulPartitionedCall?
*conv2d_transpose_5/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_4/StatefulPartitionedCall:output:01conv2d_transpose_5_statefulpartitionedcall_args_11conv2d_transpose_5_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+???????????????????????????*-
config_proto

GPU

CPU2*0J 8*V
fQRO
M__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_880192,
*conv2d_transpose_5/StatefulPartitionedCall?
&decoder_output/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_5/StatefulPartitionedCall:output:0-decoder_output_statefulpartitionedcall_args_1-decoder_output_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+???????????????????????????*-
config_proto

GPU

CPU2*0J 8*R
fMRK
I__inference_decoder_output_layer_call_and_return_conditional_losses_880622(
&decoder_output/StatefulPartitionedCall?
IdentityIdentity/decoder_output/StatefulPartitionedCall:output:0+^conv2d_transpose_3/StatefulPartitionedCall+^conv2d_transpose_4/StatefulPartitionedCall+^conv2d_transpose_5/StatefulPartitionedCall'^decoder_output/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*O
_input_shapes>
<:??????????::::::::::2X
*conv2d_transpose_3/StatefulPartitionedCall*conv2d_transpose_3/StatefulPartitionedCall2X
*conv2d_transpose_4/StatefulPartitionedCall*conv2d_transpose_4/StatefulPartitionedCall2X
*conv2d_transpose_5/StatefulPartitionedCall*conv2d_transpose_5/StatefulPartitionedCall2P
&decoder_output/StatefulPartitionedCall&decoder_output/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:- )
'
_user_specified_namedecoder_input
??
?
 __inference__wrapped_model_87707
encoder_input?
;autoencoder_encoder_conv2d_3_conv2d_readvariableop_resource@
<autoencoder_encoder_conv2d_3_biasadd_readvariableop_resource?
;autoencoder_encoder_conv2d_4_conv2d_readvariableop_resource@
<autoencoder_encoder_conv2d_4_biasadd_readvariableop_resource?
;autoencoder_encoder_conv2d_5_conv2d_readvariableop_resource@
<autoencoder_encoder_conv2d_5_biasadd_readvariableop_resourceD
@autoencoder_encoder_latent_vector_matmul_readvariableop_resourceE
Aautoencoder_encoder_latent_vector_biasadd_readvariableop_resource>
:autoencoder_decoder_dense_1_matmul_readvariableop_resource?
;autoencoder_decoder_dense_1_biasadd_readvariableop_resourceS
Oautoencoder_decoder_conv2d_transpose_3_conv2d_transpose_readvariableop_resourceJ
Fautoencoder_decoder_conv2d_transpose_3_biasadd_readvariableop_resourceS
Oautoencoder_decoder_conv2d_transpose_4_conv2d_transpose_readvariableop_resourceJ
Fautoencoder_decoder_conv2d_transpose_4_biasadd_readvariableop_resourceS
Oautoencoder_decoder_conv2d_transpose_5_conv2d_transpose_readvariableop_resourceJ
Fautoencoder_decoder_conv2d_transpose_5_biasadd_readvariableop_resourceO
Kautoencoder_decoder_decoder_output_conv2d_transpose_readvariableop_resourceF
Bautoencoder_decoder_decoder_output_biasadd_readvariableop_resource
identity??=autoencoder/decoder/conv2d_transpose_3/BiasAdd/ReadVariableOp?Fautoencoder/decoder/conv2d_transpose_3/conv2d_transpose/ReadVariableOp?=autoencoder/decoder/conv2d_transpose_4/BiasAdd/ReadVariableOp?Fautoencoder/decoder/conv2d_transpose_4/conv2d_transpose/ReadVariableOp?=autoencoder/decoder/conv2d_transpose_5/BiasAdd/ReadVariableOp?Fautoencoder/decoder/conv2d_transpose_5/conv2d_transpose/ReadVariableOp?9autoencoder/decoder/decoder_output/BiasAdd/ReadVariableOp?Bautoencoder/decoder/decoder_output/conv2d_transpose/ReadVariableOp?2autoencoder/decoder/dense_1/BiasAdd/ReadVariableOp?1autoencoder/decoder/dense_1/MatMul/ReadVariableOp?3autoencoder/encoder/conv2d_3/BiasAdd/ReadVariableOp?2autoencoder/encoder/conv2d_3/Conv2D/ReadVariableOp?3autoencoder/encoder/conv2d_4/BiasAdd/ReadVariableOp?2autoencoder/encoder/conv2d_4/Conv2D/ReadVariableOp?3autoencoder/encoder/conv2d_5/BiasAdd/ReadVariableOp?2autoencoder/encoder/conv2d_5/Conv2D/ReadVariableOp?8autoencoder/encoder/latent_vector/BiasAdd/ReadVariableOp?7autoencoder/encoder/latent_vector/MatMul/ReadVariableOp?
2autoencoder/encoder/conv2d_3/Conv2D/ReadVariableOpReadVariableOp;autoencoder_encoder_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype024
2autoencoder/encoder/conv2d_3/Conv2D/ReadVariableOp?
#autoencoder/encoder/conv2d_3/Conv2DConv2Dencoder_input:autoencoder/encoder/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2%
#autoencoder/encoder/conv2d_3/Conv2D?
3autoencoder/encoder/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp<autoencoder_encoder_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype025
3autoencoder/encoder/conv2d_3/BiasAdd/ReadVariableOp?
$autoencoder/encoder/conv2d_3/BiasAddBiasAdd,autoencoder/encoder/conv2d_3/Conv2D:output:0;autoencoder/encoder/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2&
$autoencoder/encoder/conv2d_3/BiasAdd?
!autoencoder/encoder/conv2d_3/ReluRelu-autoencoder/encoder/conv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2#
!autoencoder/encoder/conv2d_3/Relu?
2autoencoder/encoder/conv2d_4/Conv2D/ReadVariableOpReadVariableOp;autoencoder_encoder_conv2d_4_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype024
2autoencoder/encoder/conv2d_4/Conv2D/ReadVariableOp?
#autoencoder/encoder/conv2d_4/Conv2DConv2D/autoencoder/encoder/conv2d_3/Relu:activations:0:autoencoder/encoder/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2%
#autoencoder/encoder/conv2d_4/Conv2D?
3autoencoder/encoder/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp<autoencoder_encoder_conv2d_4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype025
3autoencoder/encoder/conv2d_4/BiasAdd/ReadVariableOp?
$autoencoder/encoder/conv2d_4/BiasAddBiasAdd,autoencoder/encoder/conv2d_4/Conv2D:output:0;autoencoder/encoder/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2&
$autoencoder/encoder/conv2d_4/BiasAdd?
!autoencoder/encoder/conv2d_4/ReluRelu-autoencoder/encoder/conv2d_4/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2#
!autoencoder/encoder/conv2d_4/Relu?
2autoencoder/encoder/conv2d_5/Conv2D/ReadVariableOpReadVariableOp;autoencoder_encoder_conv2d_5_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype024
2autoencoder/encoder/conv2d_5/Conv2D/ReadVariableOp?
#autoencoder/encoder/conv2d_5/Conv2DConv2D/autoencoder/encoder/conv2d_4/Relu:activations:0:autoencoder/encoder/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2%
#autoencoder/encoder/conv2d_5/Conv2D?
3autoencoder/encoder/conv2d_5/BiasAdd/ReadVariableOpReadVariableOp<autoencoder_encoder_conv2d_5_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype025
3autoencoder/encoder/conv2d_5/BiasAdd/ReadVariableOp?
$autoencoder/encoder/conv2d_5/BiasAddBiasAdd,autoencoder/encoder/conv2d_5/Conv2D:output:0;autoencoder/encoder/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2&
$autoencoder/encoder/conv2d_5/BiasAdd?
!autoencoder/encoder/conv2d_5/ReluRelu-autoencoder/encoder/conv2d_5/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2#
!autoencoder/encoder/conv2d_5/Relu?
#autoencoder/encoder/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2%
#autoencoder/encoder/flatten_1/Const?
%autoencoder/encoder/flatten_1/ReshapeReshape/autoencoder/encoder/conv2d_5/Relu:activations:0,autoencoder/encoder/flatten_1/Const:output:0*
T0*(
_output_shapes
:?????????? 2'
%autoencoder/encoder/flatten_1/Reshape?
7autoencoder/encoder/latent_vector/MatMul/ReadVariableOpReadVariableOp@autoencoder_encoder_latent_vector_matmul_readvariableop_resource* 
_output_shapes
:
? ?*
dtype029
7autoencoder/encoder/latent_vector/MatMul/ReadVariableOp?
(autoencoder/encoder/latent_vector/MatMulMatMul.autoencoder/encoder/flatten_1/Reshape:output:0?autoencoder/encoder/latent_vector/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2*
(autoencoder/encoder/latent_vector/MatMul?
8autoencoder/encoder/latent_vector/BiasAdd/ReadVariableOpReadVariableOpAautoencoder_encoder_latent_vector_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02:
8autoencoder/encoder/latent_vector/BiasAdd/ReadVariableOp?
)autoencoder/encoder/latent_vector/BiasAddBiasAdd2autoencoder/encoder/latent_vector/MatMul:product:0@autoencoder/encoder/latent_vector/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2+
)autoencoder/encoder/latent_vector/BiasAdd?
1autoencoder/decoder/dense_1/MatMul/ReadVariableOpReadVariableOp:autoencoder_decoder_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
?? *
dtype023
1autoencoder/decoder/dense_1/MatMul/ReadVariableOp?
"autoencoder/decoder/dense_1/MatMulMatMul2autoencoder/encoder/latent_vector/BiasAdd:output:09autoencoder/decoder/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????? 2$
"autoencoder/decoder/dense_1/MatMul?
2autoencoder/decoder/dense_1/BiasAdd/ReadVariableOpReadVariableOp;autoencoder_decoder_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:? *
dtype024
2autoencoder/decoder/dense_1/BiasAdd/ReadVariableOp?
#autoencoder/decoder/dense_1/BiasAddBiasAdd,autoencoder/decoder/dense_1/MatMul:product:0:autoencoder/decoder/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????? 2%
#autoencoder/decoder/dense_1/BiasAdd?
#autoencoder/decoder/reshape_1/ShapeShape,autoencoder/decoder/dense_1/BiasAdd:output:0*
T0*
_output_shapes
:2%
#autoencoder/decoder/reshape_1/Shape?
1autoencoder/decoder/reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1autoencoder/decoder/reshape_1/strided_slice/stack?
3autoencoder/decoder/reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3autoencoder/decoder/reshape_1/strided_slice/stack_1?
3autoencoder/decoder/reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3autoencoder/decoder/reshape_1/strided_slice/stack_2?
+autoencoder/decoder/reshape_1/strided_sliceStridedSlice,autoencoder/decoder/reshape_1/Shape:output:0:autoencoder/decoder/reshape_1/strided_slice/stack:output:0<autoencoder/decoder/reshape_1/strided_slice/stack_1:output:0<autoencoder/decoder/reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+autoencoder/decoder/reshape_1/strided_slice?
-autoencoder/decoder/reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2/
-autoencoder/decoder/reshape_1/Reshape/shape/1?
-autoencoder/decoder/reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2/
-autoencoder/decoder/reshape_1/Reshape/shape/2?
-autoencoder/decoder/reshape_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :?2/
-autoencoder/decoder/reshape_1/Reshape/shape/3?
+autoencoder/decoder/reshape_1/Reshape/shapePack4autoencoder/decoder/reshape_1/strided_slice:output:06autoencoder/decoder/reshape_1/Reshape/shape/1:output:06autoencoder/decoder/reshape_1/Reshape/shape/2:output:06autoencoder/decoder/reshape_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2-
+autoencoder/decoder/reshape_1/Reshape/shape?
%autoencoder/decoder/reshape_1/ReshapeReshape,autoencoder/decoder/dense_1/BiasAdd:output:04autoencoder/decoder/reshape_1/Reshape/shape:output:0*
T0*0
_output_shapes
:??????????2'
%autoencoder/decoder/reshape_1/Reshape?
,autoencoder/decoder/conv2d_transpose_3/ShapeShape.autoencoder/decoder/reshape_1/Reshape:output:0*
T0*
_output_shapes
:2.
,autoencoder/decoder/conv2d_transpose_3/Shape?
:autoencoder/decoder/conv2d_transpose_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2<
:autoencoder/decoder/conv2d_transpose_3/strided_slice/stack?
<autoencoder/decoder/conv2d_transpose_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2>
<autoencoder/decoder/conv2d_transpose_3/strided_slice/stack_1?
<autoencoder/decoder/conv2d_transpose_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2>
<autoencoder/decoder/conv2d_transpose_3/strided_slice/stack_2?
4autoencoder/decoder/conv2d_transpose_3/strided_sliceStridedSlice5autoencoder/decoder/conv2d_transpose_3/Shape:output:0Cautoencoder/decoder/conv2d_transpose_3/strided_slice/stack:output:0Eautoencoder/decoder/conv2d_transpose_3/strided_slice/stack_1:output:0Eautoencoder/decoder/conv2d_transpose_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask26
4autoencoder/decoder/conv2d_transpose_3/strided_slice?
<autoencoder/decoder/conv2d_transpose_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2>
<autoencoder/decoder/conv2d_transpose_3/strided_slice_1/stack?
>autoencoder/decoder/conv2d_transpose_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2@
>autoencoder/decoder/conv2d_transpose_3/strided_slice_1/stack_1?
>autoencoder/decoder/conv2d_transpose_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2@
>autoencoder/decoder/conv2d_transpose_3/strided_slice_1/stack_2?
6autoencoder/decoder/conv2d_transpose_3/strided_slice_1StridedSlice5autoencoder/decoder/conv2d_transpose_3/Shape:output:0Eautoencoder/decoder/conv2d_transpose_3/strided_slice_1/stack:output:0Gautoencoder/decoder/conv2d_transpose_3/strided_slice_1/stack_1:output:0Gautoencoder/decoder/conv2d_transpose_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask28
6autoencoder/decoder/conv2d_transpose_3/strided_slice_1?
<autoencoder/decoder/conv2d_transpose_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2>
<autoencoder/decoder/conv2d_transpose_3/strided_slice_2/stack?
>autoencoder/decoder/conv2d_transpose_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2@
>autoencoder/decoder/conv2d_transpose_3/strided_slice_2/stack_1?
>autoencoder/decoder/conv2d_transpose_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2@
>autoencoder/decoder/conv2d_transpose_3/strided_slice_2/stack_2?
6autoencoder/decoder/conv2d_transpose_3/strided_slice_2StridedSlice5autoencoder/decoder/conv2d_transpose_3/Shape:output:0Eautoencoder/decoder/conv2d_transpose_3/strided_slice_2/stack:output:0Gautoencoder/decoder/conv2d_transpose_3/strided_slice_2/stack_1:output:0Gautoencoder/decoder/conv2d_transpose_3/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask28
6autoencoder/decoder/conv2d_transpose_3/strided_slice_2?
,autoencoder/decoder/conv2d_transpose_3/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2.
,autoencoder/decoder/conv2d_transpose_3/mul/y?
*autoencoder/decoder/conv2d_transpose_3/mulMul?autoencoder/decoder/conv2d_transpose_3/strided_slice_1:output:05autoencoder/decoder/conv2d_transpose_3/mul/y:output:0*
T0*
_output_shapes
: 2,
*autoencoder/decoder/conv2d_transpose_3/mul?
.autoencoder/decoder/conv2d_transpose_3/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :20
.autoencoder/decoder/conv2d_transpose_3/mul_1/y?
,autoencoder/decoder/conv2d_transpose_3/mul_1Mul?autoencoder/decoder/conv2d_transpose_3/strided_slice_2:output:07autoencoder/decoder/conv2d_transpose_3/mul_1/y:output:0*
T0*
_output_shapes
: 2.
,autoencoder/decoder/conv2d_transpose_3/mul_1?
.autoencoder/decoder/conv2d_transpose_3/stack/3Const*
_output_shapes
: *
dtype0*
value	B :20
.autoencoder/decoder/conv2d_transpose_3/stack/3?
,autoencoder/decoder/conv2d_transpose_3/stackPack=autoencoder/decoder/conv2d_transpose_3/strided_slice:output:0.autoencoder/decoder/conv2d_transpose_3/mul:z:00autoencoder/decoder/conv2d_transpose_3/mul_1:z:07autoencoder/decoder/conv2d_transpose_3/stack/3:output:0*
N*
T0*
_output_shapes
:2.
,autoencoder/decoder/conv2d_transpose_3/stack?
<autoencoder/decoder/conv2d_transpose_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2>
<autoencoder/decoder/conv2d_transpose_3/strided_slice_3/stack?
>autoencoder/decoder/conv2d_transpose_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2@
>autoencoder/decoder/conv2d_transpose_3/strided_slice_3/stack_1?
>autoencoder/decoder/conv2d_transpose_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2@
>autoencoder/decoder/conv2d_transpose_3/strided_slice_3/stack_2?
6autoencoder/decoder/conv2d_transpose_3/strided_slice_3StridedSlice5autoencoder/decoder/conv2d_transpose_3/stack:output:0Eautoencoder/decoder/conv2d_transpose_3/strided_slice_3/stack:output:0Gautoencoder/decoder/conv2d_transpose_3/strided_slice_3/stack_1:output:0Gautoencoder/decoder/conv2d_transpose_3/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask28
6autoencoder/decoder/conv2d_transpose_3/strided_slice_3?
Fautoencoder/decoder/conv2d_transpose_3/conv2d_transpose/ReadVariableOpReadVariableOpOautoencoder_decoder_conv2d_transpose_3_conv2d_transpose_readvariableop_resource*'
_output_shapes
:?*
dtype02H
Fautoencoder/decoder/conv2d_transpose_3/conv2d_transpose/ReadVariableOp?
7autoencoder/decoder/conv2d_transpose_3/conv2d_transposeConv2DBackpropInput5autoencoder/decoder/conv2d_transpose_3/stack:output:0Nautoencoder/decoder/conv2d_transpose_3/conv2d_transpose/ReadVariableOp:value:0.autoencoder/decoder/reshape_1/Reshape:output:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
29
7autoencoder/decoder/conv2d_transpose_3/conv2d_transpose?
=autoencoder/decoder/conv2d_transpose_3/BiasAdd/ReadVariableOpReadVariableOpFautoencoder_decoder_conv2d_transpose_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02?
=autoencoder/decoder/conv2d_transpose_3/BiasAdd/ReadVariableOp?
.autoencoder/decoder/conv2d_transpose_3/BiasAddBiasAdd@autoencoder/decoder/conv2d_transpose_3/conv2d_transpose:output:0Eautoencoder/decoder/conv2d_transpose_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????20
.autoencoder/decoder/conv2d_transpose_3/BiasAdd?
+autoencoder/decoder/conv2d_transpose_3/ReluRelu7autoencoder/decoder/conv2d_transpose_3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2-
+autoencoder/decoder/conv2d_transpose_3/Relu?
,autoencoder/decoder/conv2d_transpose_4/ShapeShape9autoencoder/decoder/conv2d_transpose_3/Relu:activations:0*
T0*
_output_shapes
:2.
,autoencoder/decoder/conv2d_transpose_4/Shape?
:autoencoder/decoder/conv2d_transpose_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2<
:autoencoder/decoder/conv2d_transpose_4/strided_slice/stack?
<autoencoder/decoder/conv2d_transpose_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2>
<autoencoder/decoder/conv2d_transpose_4/strided_slice/stack_1?
<autoencoder/decoder/conv2d_transpose_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2>
<autoencoder/decoder/conv2d_transpose_4/strided_slice/stack_2?
4autoencoder/decoder/conv2d_transpose_4/strided_sliceStridedSlice5autoencoder/decoder/conv2d_transpose_4/Shape:output:0Cautoencoder/decoder/conv2d_transpose_4/strided_slice/stack:output:0Eautoencoder/decoder/conv2d_transpose_4/strided_slice/stack_1:output:0Eautoencoder/decoder/conv2d_transpose_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask26
4autoencoder/decoder/conv2d_transpose_4/strided_slice?
<autoencoder/decoder/conv2d_transpose_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2>
<autoencoder/decoder/conv2d_transpose_4/strided_slice_1/stack?
>autoencoder/decoder/conv2d_transpose_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2@
>autoencoder/decoder/conv2d_transpose_4/strided_slice_1/stack_1?
>autoencoder/decoder/conv2d_transpose_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2@
>autoencoder/decoder/conv2d_transpose_4/strided_slice_1/stack_2?
6autoencoder/decoder/conv2d_transpose_4/strided_slice_1StridedSlice5autoencoder/decoder/conv2d_transpose_4/Shape:output:0Eautoencoder/decoder/conv2d_transpose_4/strided_slice_1/stack:output:0Gautoencoder/decoder/conv2d_transpose_4/strided_slice_1/stack_1:output:0Gautoencoder/decoder/conv2d_transpose_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask28
6autoencoder/decoder/conv2d_transpose_4/strided_slice_1?
<autoencoder/decoder/conv2d_transpose_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2>
<autoencoder/decoder/conv2d_transpose_4/strided_slice_2/stack?
>autoencoder/decoder/conv2d_transpose_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2@
>autoencoder/decoder/conv2d_transpose_4/strided_slice_2/stack_1?
>autoencoder/decoder/conv2d_transpose_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2@
>autoencoder/decoder/conv2d_transpose_4/strided_slice_2/stack_2?
6autoencoder/decoder/conv2d_transpose_4/strided_slice_2StridedSlice5autoencoder/decoder/conv2d_transpose_4/Shape:output:0Eautoencoder/decoder/conv2d_transpose_4/strided_slice_2/stack:output:0Gautoencoder/decoder/conv2d_transpose_4/strided_slice_2/stack_1:output:0Gautoencoder/decoder/conv2d_transpose_4/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask28
6autoencoder/decoder/conv2d_transpose_4/strided_slice_2?
,autoencoder/decoder/conv2d_transpose_4/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2.
,autoencoder/decoder/conv2d_transpose_4/mul/y?
*autoencoder/decoder/conv2d_transpose_4/mulMul?autoencoder/decoder/conv2d_transpose_4/strided_slice_1:output:05autoencoder/decoder/conv2d_transpose_4/mul/y:output:0*
T0*
_output_shapes
: 2,
*autoencoder/decoder/conv2d_transpose_4/mul?
.autoencoder/decoder/conv2d_transpose_4/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :20
.autoencoder/decoder/conv2d_transpose_4/mul_1/y?
,autoencoder/decoder/conv2d_transpose_4/mul_1Mul?autoencoder/decoder/conv2d_transpose_4/strided_slice_2:output:07autoencoder/decoder/conv2d_transpose_4/mul_1/y:output:0*
T0*
_output_shapes
: 2.
,autoencoder/decoder/conv2d_transpose_4/mul_1?
.autoencoder/decoder/conv2d_transpose_4/stack/3Const*
_output_shapes
: *
dtype0*
value	B :20
.autoencoder/decoder/conv2d_transpose_4/stack/3?
,autoencoder/decoder/conv2d_transpose_4/stackPack=autoencoder/decoder/conv2d_transpose_4/strided_slice:output:0.autoencoder/decoder/conv2d_transpose_4/mul:z:00autoencoder/decoder/conv2d_transpose_4/mul_1:z:07autoencoder/decoder/conv2d_transpose_4/stack/3:output:0*
N*
T0*
_output_shapes
:2.
,autoencoder/decoder/conv2d_transpose_4/stack?
<autoencoder/decoder/conv2d_transpose_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2>
<autoencoder/decoder/conv2d_transpose_4/strided_slice_3/stack?
>autoencoder/decoder/conv2d_transpose_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2@
>autoencoder/decoder/conv2d_transpose_4/strided_slice_3/stack_1?
>autoencoder/decoder/conv2d_transpose_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2@
>autoencoder/decoder/conv2d_transpose_4/strided_slice_3/stack_2?
6autoencoder/decoder/conv2d_transpose_4/strided_slice_3StridedSlice5autoencoder/decoder/conv2d_transpose_4/stack:output:0Eautoencoder/decoder/conv2d_transpose_4/strided_slice_3/stack:output:0Gautoencoder/decoder/conv2d_transpose_4/strided_slice_3/stack_1:output:0Gautoencoder/decoder/conv2d_transpose_4/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask28
6autoencoder/decoder/conv2d_transpose_4/strided_slice_3?
Fautoencoder/decoder/conv2d_transpose_4/conv2d_transpose/ReadVariableOpReadVariableOpOautoencoder_decoder_conv2d_transpose_4_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02H
Fautoencoder/decoder/conv2d_transpose_4/conv2d_transpose/ReadVariableOp?
7autoencoder/decoder/conv2d_transpose_4/conv2d_transposeConv2DBackpropInput5autoencoder/decoder/conv2d_transpose_4/stack:output:0Nautoencoder/decoder/conv2d_transpose_4/conv2d_transpose/ReadVariableOp:value:09autoencoder/decoder/conv2d_transpose_3/Relu:activations:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
29
7autoencoder/decoder/conv2d_transpose_4/conv2d_transpose?
=autoencoder/decoder/conv2d_transpose_4/BiasAdd/ReadVariableOpReadVariableOpFautoencoder_decoder_conv2d_transpose_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02?
=autoencoder/decoder/conv2d_transpose_4/BiasAdd/ReadVariableOp?
.autoencoder/decoder/conv2d_transpose_4/BiasAddBiasAdd@autoencoder/decoder/conv2d_transpose_4/conv2d_transpose:output:0Eautoencoder/decoder/conv2d_transpose_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????20
.autoencoder/decoder/conv2d_transpose_4/BiasAdd?
+autoencoder/decoder/conv2d_transpose_4/ReluRelu7autoencoder/decoder/conv2d_transpose_4/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2-
+autoencoder/decoder/conv2d_transpose_4/Relu?
,autoencoder/decoder/conv2d_transpose_5/ShapeShape9autoencoder/decoder/conv2d_transpose_4/Relu:activations:0*
T0*
_output_shapes
:2.
,autoencoder/decoder/conv2d_transpose_5/Shape?
:autoencoder/decoder/conv2d_transpose_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2<
:autoencoder/decoder/conv2d_transpose_5/strided_slice/stack?
<autoencoder/decoder/conv2d_transpose_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2>
<autoencoder/decoder/conv2d_transpose_5/strided_slice/stack_1?
<autoencoder/decoder/conv2d_transpose_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2>
<autoencoder/decoder/conv2d_transpose_5/strided_slice/stack_2?
4autoencoder/decoder/conv2d_transpose_5/strided_sliceStridedSlice5autoencoder/decoder/conv2d_transpose_5/Shape:output:0Cautoencoder/decoder/conv2d_transpose_5/strided_slice/stack:output:0Eautoencoder/decoder/conv2d_transpose_5/strided_slice/stack_1:output:0Eautoencoder/decoder/conv2d_transpose_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask26
4autoencoder/decoder/conv2d_transpose_5/strided_slice?
<autoencoder/decoder/conv2d_transpose_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2>
<autoencoder/decoder/conv2d_transpose_5/strided_slice_1/stack?
>autoencoder/decoder/conv2d_transpose_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2@
>autoencoder/decoder/conv2d_transpose_5/strided_slice_1/stack_1?
>autoencoder/decoder/conv2d_transpose_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2@
>autoencoder/decoder/conv2d_transpose_5/strided_slice_1/stack_2?
6autoencoder/decoder/conv2d_transpose_5/strided_slice_1StridedSlice5autoencoder/decoder/conv2d_transpose_5/Shape:output:0Eautoencoder/decoder/conv2d_transpose_5/strided_slice_1/stack:output:0Gautoencoder/decoder/conv2d_transpose_5/strided_slice_1/stack_1:output:0Gautoencoder/decoder/conv2d_transpose_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask28
6autoencoder/decoder/conv2d_transpose_5/strided_slice_1?
<autoencoder/decoder/conv2d_transpose_5/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2>
<autoencoder/decoder/conv2d_transpose_5/strided_slice_2/stack?
>autoencoder/decoder/conv2d_transpose_5/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2@
>autoencoder/decoder/conv2d_transpose_5/strided_slice_2/stack_1?
>autoencoder/decoder/conv2d_transpose_5/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2@
>autoencoder/decoder/conv2d_transpose_5/strided_slice_2/stack_2?
6autoencoder/decoder/conv2d_transpose_5/strided_slice_2StridedSlice5autoencoder/decoder/conv2d_transpose_5/Shape:output:0Eautoencoder/decoder/conv2d_transpose_5/strided_slice_2/stack:output:0Gautoencoder/decoder/conv2d_transpose_5/strided_slice_2/stack_1:output:0Gautoencoder/decoder/conv2d_transpose_5/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask28
6autoencoder/decoder/conv2d_transpose_5/strided_slice_2?
,autoencoder/decoder/conv2d_transpose_5/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2.
,autoencoder/decoder/conv2d_transpose_5/mul/y?
*autoencoder/decoder/conv2d_transpose_5/mulMul?autoencoder/decoder/conv2d_transpose_5/strided_slice_1:output:05autoencoder/decoder/conv2d_transpose_5/mul/y:output:0*
T0*
_output_shapes
: 2,
*autoencoder/decoder/conv2d_transpose_5/mul?
.autoencoder/decoder/conv2d_transpose_5/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :20
.autoencoder/decoder/conv2d_transpose_5/mul_1/y?
,autoencoder/decoder/conv2d_transpose_5/mul_1Mul?autoencoder/decoder/conv2d_transpose_5/strided_slice_2:output:07autoencoder/decoder/conv2d_transpose_5/mul_1/y:output:0*
T0*
_output_shapes
: 2.
,autoencoder/decoder/conv2d_transpose_5/mul_1?
.autoencoder/decoder/conv2d_transpose_5/stack/3Const*
_output_shapes
: *
dtype0*
value	B :20
.autoencoder/decoder/conv2d_transpose_5/stack/3?
,autoencoder/decoder/conv2d_transpose_5/stackPack=autoencoder/decoder/conv2d_transpose_5/strided_slice:output:0.autoencoder/decoder/conv2d_transpose_5/mul:z:00autoencoder/decoder/conv2d_transpose_5/mul_1:z:07autoencoder/decoder/conv2d_transpose_5/stack/3:output:0*
N*
T0*
_output_shapes
:2.
,autoencoder/decoder/conv2d_transpose_5/stack?
<autoencoder/decoder/conv2d_transpose_5/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2>
<autoencoder/decoder/conv2d_transpose_5/strided_slice_3/stack?
>autoencoder/decoder/conv2d_transpose_5/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2@
>autoencoder/decoder/conv2d_transpose_5/strided_slice_3/stack_1?
>autoencoder/decoder/conv2d_transpose_5/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2@
>autoencoder/decoder/conv2d_transpose_5/strided_slice_3/stack_2?
6autoencoder/decoder/conv2d_transpose_5/strided_slice_3StridedSlice5autoencoder/decoder/conv2d_transpose_5/stack:output:0Eautoencoder/decoder/conv2d_transpose_5/strided_slice_3/stack:output:0Gautoencoder/decoder/conv2d_transpose_5/strided_slice_3/stack_1:output:0Gautoencoder/decoder/conv2d_transpose_5/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask28
6autoencoder/decoder/conv2d_transpose_5/strided_slice_3?
Fautoencoder/decoder/conv2d_transpose_5/conv2d_transpose/ReadVariableOpReadVariableOpOautoencoder_decoder_conv2d_transpose_5_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02H
Fautoencoder/decoder/conv2d_transpose_5/conv2d_transpose/ReadVariableOp?
7autoencoder/decoder/conv2d_transpose_5/conv2d_transposeConv2DBackpropInput5autoencoder/decoder/conv2d_transpose_5/stack:output:0Nautoencoder/decoder/conv2d_transpose_5/conv2d_transpose/ReadVariableOp:value:09autoencoder/decoder/conv2d_transpose_4/Relu:activations:0*
T0*/
_output_shapes
:?????????  *
paddingSAME*
strides
29
7autoencoder/decoder/conv2d_transpose_5/conv2d_transpose?
=autoencoder/decoder/conv2d_transpose_5/BiasAdd/ReadVariableOpReadVariableOpFautoencoder_decoder_conv2d_transpose_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02?
=autoencoder/decoder/conv2d_transpose_5/BiasAdd/ReadVariableOp?
.autoencoder/decoder/conv2d_transpose_5/BiasAddBiasAdd@autoencoder/decoder/conv2d_transpose_5/conv2d_transpose:output:0Eautoencoder/decoder/conv2d_transpose_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  20
.autoencoder/decoder/conv2d_transpose_5/BiasAdd?
+autoencoder/decoder/conv2d_transpose_5/ReluRelu7autoencoder/decoder/conv2d_transpose_5/BiasAdd:output:0*
T0*/
_output_shapes
:?????????  2-
+autoencoder/decoder/conv2d_transpose_5/Relu?
(autoencoder/decoder/decoder_output/ShapeShape9autoencoder/decoder/conv2d_transpose_5/Relu:activations:0*
T0*
_output_shapes
:2*
(autoencoder/decoder/decoder_output/Shape?
6autoencoder/decoder/decoder_output/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 28
6autoencoder/decoder/decoder_output/strided_slice/stack?
8autoencoder/decoder/decoder_output/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2:
8autoencoder/decoder/decoder_output/strided_slice/stack_1?
8autoencoder/decoder/decoder_output/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2:
8autoencoder/decoder/decoder_output/strided_slice/stack_2?
0autoencoder/decoder/decoder_output/strided_sliceStridedSlice1autoencoder/decoder/decoder_output/Shape:output:0?autoencoder/decoder/decoder_output/strided_slice/stack:output:0Aautoencoder/decoder/decoder_output/strided_slice/stack_1:output:0Aautoencoder/decoder/decoder_output/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask22
0autoencoder/decoder/decoder_output/strided_slice?
8autoencoder/decoder/decoder_output/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2:
8autoencoder/decoder/decoder_output/strided_slice_1/stack?
:autoencoder/decoder/decoder_output/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2<
:autoencoder/decoder/decoder_output/strided_slice_1/stack_1?
:autoencoder/decoder/decoder_output/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2<
:autoencoder/decoder/decoder_output/strided_slice_1/stack_2?
2autoencoder/decoder/decoder_output/strided_slice_1StridedSlice1autoencoder/decoder/decoder_output/Shape:output:0Aautoencoder/decoder/decoder_output/strided_slice_1/stack:output:0Cautoencoder/decoder/decoder_output/strided_slice_1/stack_1:output:0Cautoencoder/decoder/decoder_output/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask24
2autoencoder/decoder/decoder_output/strided_slice_1?
8autoencoder/decoder/decoder_output/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2:
8autoencoder/decoder/decoder_output/strided_slice_2/stack?
:autoencoder/decoder/decoder_output/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2<
:autoencoder/decoder/decoder_output/strided_slice_2/stack_1?
:autoencoder/decoder/decoder_output/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2<
:autoencoder/decoder/decoder_output/strided_slice_2/stack_2?
2autoencoder/decoder/decoder_output/strided_slice_2StridedSlice1autoencoder/decoder/decoder_output/Shape:output:0Aautoencoder/decoder/decoder_output/strided_slice_2/stack:output:0Cautoencoder/decoder/decoder_output/strided_slice_2/stack_1:output:0Cautoencoder/decoder/decoder_output/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask24
2autoencoder/decoder/decoder_output/strided_slice_2?
(autoencoder/decoder/decoder_output/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2*
(autoencoder/decoder/decoder_output/mul/y?
&autoencoder/decoder/decoder_output/mulMul;autoencoder/decoder/decoder_output/strided_slice_1:output:01autoencoder/decoder/decoder_output/mul/y:output:0*
T0*
_output_shapes
: 2(
&autoencoder/decoder/decoder_output/mul?
*autoencoder/decoder/decoder_output/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2,
*autoencoder/decoder/decoder_output/mul_1/y?
(autoencoder/decoder/decoder_output/mul_1Mul;autoencoder/decoder/decoder_output/strided_slice_2:output:03autoencoder/decoder/decoder_output/mul_1/y:output:0*
T0*
_output_shapes
: 2*
(autoencoder/decoder/decoder_output/mul_1?
*autoencoder/decoder/decoder_output/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2,
*autoencoder/decoder/decoder_output/stack/3?
(autoencoder/decoder/decoder_output/stackPack9autoencoder/decoder/decoder_output/strided_slice:output:0*autoencoder/decoder/decoder_output/mul:z:0,autoencoder/decoder/decoder_output/mul_1:z:03autoencoder/decoder/decoder_output/stack/3:output:0*
N*
T0*
_output_shapes
:2*
(autoencoder/decoder/decoder_output/stack?
8autoencoder/decoder/decoder_output/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2:
8autoencoder/decoder/decoder_output/strided_slice_3/stack?
:autoencoder/decoder/decoder_output/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2<
:autoencoder/decoder/decoder_output/strided_slice_3/stack_1?
:autoencoder/decoder/decoder_output/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2<
:autoencoder/decoder/decoder_output/strided_slice_3/stack_2?
2autoencoder/decoder/decoder_output/strided_slice_3StridedSlice1autoencoder/decoder/decoder_output/stack:output:0Aautoencoder/decoder/decoder_output/strided_slice_3/stack:output:0Cautoencoder/decoder/decoder_output/strided_slice_3/stack_1:output:0Cautoencoder/decoder/decoder_output/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask24
2autoencoder/decoder/decoder_output/strided_slice_3?
Bautoencoder/decoder/decoder_output/conv2d_transpose/ReadVariableOpReadVariableOpKautoencoder_decoder_decoder_output_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02D
Bautoencoder/decoder/decoder_output/conv2d_transpose/ReadVariableOp?
3autoencoder/decoder/decoder_output/conv2d_transposeConv2DBackpropInput1autoencoder/decoder/decoder_output/stack:output:0Jautoencoder/decoder/decoder_output/conv2d_transpose/ReadVariableOp:value:09autoencoder/decoder/conv2d_transpose_5/Relu:activations:0*
T0*/
_output_shapes
:?????????  *
paddingSAME*
strides
25
3autoencoder/decoder/decoder_output/conv2d_transpose?
9autoencoder/decoder/decoder_output/BiasAdd/ReadVariableOpReadVariableOpBautoencoder_decoder_decoder_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02;
9autoencoder/decoder/decoder_output/BiasAdd/ReadVariableOp?
*autoencoder/decoder/decoder_output/BiasAddBiasAdd<autoencoder/decoder/decoder_output/conv2d_transpose:output:0Aautoencoder/decoder/decoder_output/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  2,
*autoencoder/decoder/decoder_output/BiasAdd?
*autoencoder/decoder/decoder_output/SigmoidSigmoid3autoencoder/decoder/decoder_output/BiasAdd:output:0*
T0*/
_output_shapes
:?????????  2,
*autoencoder/decoder/decoder_output/Sigmoid?	
IdentityIdentity.autoencoder/decoder/decoder_output/Sigmoid:y:0>^autoencoder/decoder/conv2d_transpose_3/BiasAdd/ReadVariableOpG^autoencoder/decoder/conv2d_transpose_3/conv2d_transpose/ReadVariableOp>^autoencoder/decoder/conv2d_transpose_4/BiasAdd/ReadVariableOpG^autoencoder/decoder/conv2d_transpose_4/conv2d_transpose/ReadVariableOp>^autoencoder/decoder/conv2d_transpose_5/BiasAdd/ReadVariableOpG^autoencoder/decoder/conv2d_transpose_5/conv2d_transpose/ReadVariableOp:^autoencoder/decoder/decoder_output/BiasAdd/ReadVariableOpC^autoencoder/decoder/decoder_output/conv2d_transpose/ReadVariableOp3^autoencoder/decoder/dense_1/BiasAdd/ReadVariableOp2^autoencoder/decoder/dense_1/MatMul/ReadVariableOp4^autoencoder/encoder/conv2d_3/BiasAdd/ReadVariableOp3^autoencoder/encoder/conv2d_3/Conv2D/ReadVariableOp4^autoencoder/encoder/conv2d_4/BiasAdd/ReadVariableOp3^autoencoder/encoder/conv2d_4/Conv2D/ReadVariableOp4^autoencoder/encoder/conv2d_5/BiasAdd/ReadVariableOp3^autoencoder/encoder/conv2d_5/Conv2D/ReadVariableOp9^autoencoder/encoder/latent_vector/BiasAdd/ReadVariableOp8^autoencoder/encoder/latent_vector/MatMul/ReadVariableOp*
T0*/
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*v
_input_shapese
c:?????????  ::::::::::::::::::2~
=autoencoder/decoder/conv2d_transpose_3/BiasAdd/ReadVariableOp=autoencoder/decoder/conv2d_transpose_3/BiasAdd/ReadVariableOp2?
Fautoencoder/decoder/conv2d_transpose_3/conv2d_transpose/ReadVariableOpFautoencoder/decoder/conv2d_transpose_3/conv2d_transpose/ReadVariableOp2~
=autoencoder/decoder/conv2d_transpose_4/BiasAdd/ReadVariableOp=autoencoder/decoder/conv2d_transpose_4/BiasAdd/ReadVariableOp2?
Fautoencoder/decoder/conv2d_transpose_4/conv2d_transpose/ReadVariableOpFautoencoder/decoder/conv2d_transpose_4/conv2d_transpose/ReadVariableOp2~
=autoencoder/decoder/conv2d_transpose_5/BiasAdd/ReadVariableOp=autoencoder/decoder/conv2d_transpose_5/BiasAdd/ReadVariableOp2?
Fautoencoder/decoder/conv2d_transpose_5/conv2d_transpose/ReadVariableOpFautoencoder/decoder/conv2d_transpose_5/conv2d_transpose/ReadVariableOp2v
9autoencoder/decoder/decoder_output/BiasAdd/ReadVariableOp9autoencoder/decoder/decoder_output/BiasAdd/ReadVariableOp2?
Bautoencoder/decoder/decoder_output/conv2d_transpose/ReadVariableOpBautoencoder/decoder/decoder_output/conv2d_transpose/ReadVariableOp2h
2autoencoder/decoder/dense_1/BiasAdd/ReadVariableOp2autoencoder/decoder/dense_1/BiasAdd/ReadVariableOp2f
1autoencoder/decoder/dense_1/MatMul/ReadVariableOp1autoencoder/decoder/dense_1/MatMul/ReadVariableOp2j
3autoencoder/encoder/conv2d_3/BiasAdd/ReadVariableOp3autoencoder/encoder/conv2d_3/BiasAdd/ReadVariableOp2h
2autoencoder/encoder/conv2d_3/Conv2D/ReadVariableOp2autoencoder/encoder/conv2d_3/Conv2D/ReadVariableOp2j
3autoencoder/encoder/conv2d_4/BiasAdd/ReadVariableOp3autoencoder/encoder/conv2d_4/BiasAdd/ReadVariableOp2h
2autoencoder/encoder/conv2d_4/Conv2D/ReadVariableOp2autoencoder/encoder/conv2d_4/Conv2D/ReadVariableOp2j
3autoencoder/encoder/conv2d_5/BiasAdd/ReadVariableOp3autoencoder/encoder/conv2d_5/BiasAdd/ReadVariableOp2h
2autoencoder/encoder/conv2d_5/Conv2D/ReadVariableOp2autoencoder/encoder/conv2d_5/Conv2D/ReadVariableOp2t
8autoencoder/encoder/latent_vector/BiasAdd/ReadVariableOp8autoencoder/encoder/latent_vector/BiasAdd/ReadVariableOp2r
7autoencoder/encoder/latent_vector/MatMul/ReadVariableOp7autoencoder/encoder/latent_vector/MatMul/ReadVariableOp:- )
'
_user_specified_nameencoder_input
?
?
(__inference_conv2d_4_layer_call_fn_87749

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,????????????????????????????*-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_conv2d_4_layer_call_and_return_conditional_losses_877412
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????@::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?
`
D__inference_flatten_1_layer_call_and_return_conditional_losses_87789

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:?????????? 2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:?????????? 2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:& "
 
_user_specified_nameinputs
?%
?
B__inference_decoder_layer_call_and_return_conditional_losses_88174

inputs*
&dense_1_statefulpartitionedcall_args_1*
&dense_1_statefulpartitionedcall_args_25
1conv2d_transpose_3_statefulpartitionedcall_args_15
1conv2d_transpose_3_statefulpartitionedcall_args_25
1conv2d_transpose_4_statefulpartitionedcall_args_15
1conv2d_transpose_4_statefulpartitionedcall_args_25
1conv2d_transpose_5_statefulpartitionedcall_args_15
1conv2d_transpose_5_statefulpartitionedcall_args_21
-decoder_output_statefulpartitionedcall_args_11
-decoder_output_statefulpartitionedcall_args_2
identity??*conv2d_transpose_3/StatefulPartitionedCall?*conv2d_transpose_4/StatefulPartitionedCall?*conv2d_transpose_5/StatefulPartitionedCall?&decoder_output/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCallinputs&dense_1_statefulpartitionedcall_args_1&dense_1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:?????????? *-
config_proto

GPU

CPU2*0J 8*K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_880842!
dense_1/StatefulPartitionedCall?
reshape_1/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:??????????*-
config_proto

GPU

CPU2*0J 8*M
fHRF
D__inference_reshape_1_layer_call_and_return_conditional_losses_881102
reshape_1/PartitionedCall?
*conv2d_transpose_3/StatefulPartitionedCallStatefulPartitionedCall"reshape_1/PartitionedCall:output:01conv2d_transpose_3_statefulpartitionedcall_args_11conv2d_transpose_3_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+???????????????????????????*-
config_proto

GPU

CPU2*0J 8*V
fQRO
M__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_879332,
*conv2d_transpose_3/StatefulPartitionedCall?
*conv2d_transpose_4/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_3/StatefulPartitionedCall:output:01conv2d_transpose_4_statefulpartitionedcall_args_11conv2d_transpose_4_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+???????????????????????????*-
config_proto

GPU

CPU2*0J 8*V
fQRO
M__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_879762,
*conv2d_transpose_4/StatefulPartitionedCall?
*conv2d_transpose_5/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_4/StatefulPartitionedCall:output:01conv2d_transpose_5_statefulpartitionedcall_args_11conv2d_transpose_5_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+???????????????????????????*-
config_proto

GPU

CPU2*0J 8*V
fQRO
M__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_880192,
*conv2d_transpose_5/StatefulPartitionedCall?
&decoder_output/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_5/StatefulPartitionedCall:output:0-decoder_output_statefulpartitionedcall_args_1-decoder_output_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+???????????????????????????*-
config_proto

GPU

CPU2*0J 8*R
fMRK
I__inference_decoder_output_layer_call_and_return_conditional_losses_880622(
&decoder_output/StatefulPartitionedCall?
IdentityIdentity/decoder_output/StatefulPartitionedCall:output:0+^conv2d_transpose_3/StatefulPartitionedCall+^conv2d_transpose_4/StatefulPartitionedCall+^conv2d_transpose_5/StatefulPartitionedCall'^decoder_output/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*O
_input_shapes>
<:??????????::::::::::2X
*conv2d_transpose_3/StatefulPartitionedCall*conv2d_transpose_3/StatefulPartitionedCall2X
*conv2d_transpose_4/StatefulPartitionedCall*conv2d_transpose_4/StatefulPartitionedCall2X
*conv2d_transpose_5/StatefulPartitionedCall*conv2d_transpose_5/StatefulPartitionedCall2P
&decoder_output/StatefulPartitionedCall&decoder_output/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:& "
 
_user_specified_nameinputs
?
?
+__inference_autoencoder_layer_call_fn_88375
encoder_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallencoder_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+???????????????????????????*-
config_proto

GPU

CPU2*0J 8*O
fJRH
F__inference_autoencoder_layer_call_and_return_conditional_losses_883542
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*v
_input_shapese
c:?????????  ::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:- )
'
_user_specified_nameencoder_input
?i
?
__inference__traced_save_89505
file_prefix(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop.
*savev2_conv2d_3_kernel_read_readvariableop,
(savev2_conv2d_3_bias_read_readvariableop.
*savev2_conv2d_4_kernel_read_readvariableop,
(savev2_conv2d_4_bias_read_readvariableop.
*savev2_conv2d_5_kernel_read_readvariableop,
(savev2_conv2d_5_bias_read_readvariableop5
1savev2_latent_vector_1_kernel_read_readvariableop3
/savev2_latent_vector_1_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop8
4savev2_conv2d_transpose_3_kernel_read_readvariableop6
2savev2_conv2d_transpose_3_bias_read_readvariableop8
4savev2_conv2d_transpose_4_kernel_read_readvariableop6
2savev2_conv2d_transpose_4_bias_read_readvariableop8
4savev2_conv2d_transpose_5_kernel_read_readvariableop6
2savev2_conv2d_transpose_5_bias_read_readvariableop6
2savev2_decoder_output_1_kernel_read_readvariableop4
0savev2_decoder_output_1_bias_read_readvariableop5
1savev2_adam_conv2d_3_kernel_m_read_readvariableop3
/savev2_adam_conv2d_3_bias_m_read_readvariableop5
1savev2_adam_conv2d_4_kernel_m_read_readvariableop3
/savev2_adam_conv2d_4_bias_m_read_readvariableop5
1savev2_adam_conv2d_5_kernel_m_read_readvariableop3
/savev2_adam_conv2d_5_bias_m_read_readvariableop<
8savev2_adam_latent_vector_1_kernel_m_read_readvariableop:
6savev2_adam_latent_vector_1_bias_m_read_readvariableop4
0savev2_adam_dense_1_kernel_m_read_readvariableop2
.savev2_adam_dense_1_bias_m_read_readvariableop?
;savev2_adam_conv2d_transpose_3_kernel_m_read_readvariableop=
9savev2_adam_conv2d_transpose_3_bias_m_read_readvariableop?
;savev2_adam_conv2d_transpose_4_kernel_m_read_readvariableop=
9savev2_adam_conv2d_transpose_4_bias_m_read_readvariableop?
;savev2_adam_conv2d_transpose_5_kernel_m_read_readvariableop=
9savev2_adam_conv2d_transpose_5_bias_m_read_readvariableop=
9savev2_adam_decoder_output_1_kernel_m_read_readvariableop;
7savev2_adam_decoder_output_1_bias_m_read_readvariableop5
1savev2_adam_conv2d_3_kernel_v_read_readvariableop3
/savev2_adam_conv2d_3_bias_v_read_readvariableop5
1savev2_adam_conv2d_4_kernel_v_read_readvariableop3
/savev2_adam_conv2d_4_bias_v_read_readvariableop5
1savev2_adam_conv2d_5_kernel_v_read_readvariableop3
/savev2_adam_conv2d_5_bias_v_read_readvariableop<
8savev2_adam_latent_vector_1_kernel_v_read_readvariableop:
6savev2_adam_latent_vector_1_bias_v_read_readvariableop4
0savev2_adam_dense_1_kernel_v_read_readvariableop2
.savev2_adam_dense_1_bias_v_read_readvariableop?
;savev2_adam_conv2d_transpose_3_kernel_v_read_readvariableop=
9savev2_adam_conv2d_transpose_3_bias_v_read_readvariableop?
;savev2_adam_conv2d_transpose_4_kernel_v_read_readvariableop=
9savev2_adam_conv2d_transpose_4_bias_v_read_readvariableop?
;savev2_adam_conv2d_transpose_5_kernel_v_read_readvariableop=
9savev2_adam_conv2d_transpose_5_bias_v_read_readvariableop=
9savev2_adam_decoder_output_1_kernel_v_read_readvariableop;
7savev2_adam_decoder_output_1_bias_v_read_readvariableop
savev2_1_const

identity_1??MergeV2Checkpoints?SaveV2?SaveV2_1?
StringJoin/inputs_1Const"/device:CPU:0*
_output_shapes
: *
dtype0*<
value3B1 B+_temp_f14092bcbf89454d8f0bbaadb1db8e03/part2
StringJoin/inputs_1?

StringJoin
StringJoinfile_prefixStringJoin/inputs_1:output:0"/device:CPU:0*
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:;*
dtype0*?
value?B?;B)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:;*
dtype0*?
value?B~;B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop*savev2_conv2d_3_kernel_read_readvariableop(savev2_conv2d_3_bias_read_readvariableop*savev2_conv2d_4_kernel_read_readvariableop(savev2_conv2d_4_bias_read_readvariableop*savev2_conv2d_5_kernel_read_readvariableop(savev2_conv2d_5_bias_read_readvariableop1savev2_latent_vector_1_kernel_read_readvariableop/savev2_latent_vector_1_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop4savev2_conv2d_transpose_3_kernel_read_readvariableop2savev2_conv2d_transpose_3_bias_read_readvariableop4savev2_conv2d_transpose_4_kernel_read_readvariableop2savev2_conv2d_transpose_4_bias_read_readvariableop4savev2_conv2d_transpose_5_kernel_read_readvariableop2savev2_conv2d_transpose_5_bias_read_readvariableop2savev2_decoder_output_1_kernel_read_readvariableop0savev2_decoder_output_1_bias_read_readvariableop1savev2_adam_conv2d_3_kernel_m_read_readvariableop/savev2_adam_conv2d_3_bias_m_read_readvariableop1savev2_adam_conv2d_4_kernel_m_read_readvariableop/savev2_adam_conv2d_4_bias_m_read_readvariableop1savev2_adam_conv2d_5_kernel_m_read_readvariableop/savev2_adam_conv2d_5_bias_m_read_readvariableop8savev2_adam_latent_vector_1_kernel_m_read_readvariableop6savev2_adam_latent_vector_1_bias_m_read_readvariableop0savev2_adam_dense_1_kernel_m_read_readvariableop.savev2_adam_dense_1_bias_m_read_readvariableop;savev2_adam_conv2d_transpose_3_kernel_m_read_readvariableop9savev2_adam_conv2d_transpose_3_bias_m_read_readvariableop;savev2_adam_conv2d_transpose_4_kernel_m_read_readvariableop9savev2_adam_conv2d_transpose_4_bias_m_read_readvariableop;savev2_adam_conv2d_transpose_5_kernel_m_read_readvariableop9savev2_adam_conv2d_transpose_5_bias_m_read_readvariableop9savev2_adam_decoder_output_1_kernel_m_read_readvariableop7savev2_adam_decoder_output_1_bias_m_read_readvariableop1savev2_adam_conv2d_3_kernel_v_read_readvariableop/savev2_adam_conv2d_3_bias_v_read_readvariableop1savev2_adam_conv2d_4_kernel_v_read_readvariableop/savev2_adam_conv2d_4_bias_v_read_readvariableop1savev2_adam_conv2d_5_kernel_v_read_readvariableop/savev2_adam_conv2d_5_bias_v_read_readvariableop8savev2_adam_latent_vector_1_kernel_v_read_readvariableop6savev2_adam_latent_vector_1_bias_v_read_readvariableop0savev2_adam_dense_1_kernel_v_read_readvariableop.savev2_adam_dense_1_bias_v_read_readvariableop;savev2_adam_conv2d_transpose_3_kernel_v_read_readvariableop9savev2_adam_conv2d_transpose_3_bias_v_read_readvariableop;savev2_adam_conv2d_transpose_4_kernel_v_read_readvariableop9savev2_adam_conv2d_transpose_4_bias_v_read_readvariableop;savev2_adam_conv2d_transpose_5_kernel_v_read_readvariableop9savev2_adam_conv2d_transpose_5_bias_v_read_readvariableop9savev2_adam_decoder_output_1_kernel_v_read_readvariableop7savev2_adam_decoder_output_1_bias_v_read_readvariableop"/device:CPU:0*
_output_shapes
 *I
dtypes?
=2;	2
SaveV2?
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :2
ShardedFilename_1/shard?
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename_1?
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2_1/tensor_names?
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2_1/shape_and_slices?
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
22

SaveV2_1?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity?

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: : : : : : :@:@:@?:?:??:?:
? ?:?:
?? :? :?::::::::@:@:@?:?:??:?:
? ?:?:
?? :? :?::::::::@:@:@?:?:??:?:
? ?:?:
?? :? :?:::::::: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:+ '
%
_user_specified_namefile_prefix
?
?
2__inference_conv2d_transpose_5_layer_call_fn_88027

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+???????????????????????????*-
config_proto

GPU

CPU2*0J 8*V
fQRO
M__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_880192
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?

?
'__inference_encoder_layer_call_fn_88930

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8*
Tin
2	*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????*-
config_proto

GPU

CPU2*0J 8*K
fFRD
B__inference_encoder_layer_call_and_return_conditional_losses_878872
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????  ::::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?
?
C__inference_conv2d_4_layer_call_and_return_conditional_losses_87741

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOpo
dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
dilation_rate?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????2	
BiasAdds
ReluReluBiasAdd:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:& "
 
_user_specified_nameinputs
?*
?
B__inference_encoder_layer_call_and_return_conditional_losses_88871

inputs+
'conv2d_3_conv2d_readvariableop_resource,
(conv2d_3_biasadd_readvariableop_resource+
'conv2d_4_conv2d_readvariableop_resource,
(conv2d_4_biasadd_readvariableop_resource+
'conv2d_5_conv2d_readvariableop_resource,
(conv2d_5_biasadd_readvariableop_resource0
,latent_vector_matmul_readvariableop_resource1
-latent_vector_biasadd_readvariableop_resource
identity??conv2d_3/BiasAdd/ReadVariableOp?conv2d_3/Conv2D/ReadVariableOp?conv2d_4/BiasAdd/ReadVariableOp?conv2d_4/Conv2D/ReadVariableOp?conv2d_5/BiasAdd/ReadVariableOp?conv2d_5/Conv2D/ReadVariableOp?$latent_vector/BiasAdd/ReadVariableOp?#latent_vector/MatMul/ReadVariableOp?
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02 
conv2d_3/Conv2D/ReadVariableOp?
conv2d_3/Conv2DConv2Dinputs&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
conv2d_3/Conv2D?
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_3/BiasAdd/ReadVariableOp?
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_3/BiasAdd{
conv2d_3/ReluReluconv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
conv2d_3/Relu?
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02 
conv2d_4/Conv2D/ReadVariableOp?
conv2d_4/Conv2DConv2Dconv2d_3/Relu:activations:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv2d_4/Conv2D?
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
conv2d_4/BiasAdd/ReadVariableOp?
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_4/BiasAdd|
conv2d_4/ReluReluconv2d_4/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
conv2d_4/Relu?
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02 
conv2d_5/Conv2D/ReadVariableOp?
conv2d_5/Conv2DConv2Dconv2d_4/Relu:activations:0&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv2d_5/Conv2D?
conv2d_5/BiasAdd/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
conv2d_5/BiasAdd/ReadVariableOp?
conv2d_5/BiasAddBiasAddconv2d_5/Conv2D:output:0'conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_5/BiasAdd|
conv2d_5/ReluReluconv2d_5/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
conv2d_5/Relus
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_1/Const?
flatten_1/ReshapeReshapeconv2d_5/Relu:activations:0flatten_1/Const:output:0*
T0*(
_output_shapes
:?????????? 2
flatten_1/Reshape?
#latent_vector/MatMul/ReadVariableOpReadVariableOp,latent_vector_matmul_readvariableop_resource* 
_output_shapes
:
? ?*
dtype02%
#latent_vector/MatMul/ReadVariableOp?
latent_vector/MatMulMatMulflatten_1/Reshape:output:0+latent_vector/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
latent_vector/MatMul?
$latent_vector/BiasAdd/ReadVariableOpReadVariableOp-latent_vector_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$latent_vector/BiasAdd/ReadVariableOp?
latent_vector/BiasAddBiasAddlatent_vector/MatMul:product:0,latent_vector/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
latent_vector/BiasAdd?
IdentityIdentitylatent_vector/BiasAdd:output:0 ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp ^conv2d_5/BiasAdd/ReadVariableOp^conv2d_5/Conv2D/ReadVariableOp%^latent_vector/BiasAdd/ReadVariableOp$^latent_vector/MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????  ::::::::2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2B
conv2d_4/BiasAdd/ReadVariableOpconv2d_4/BiasAdd/ReadVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp2B
conv2d_5/BiasAdd/ReadVariableOpconv2d_5/BiasAdd/ReadVariableOp2@
conv2d_5/Conv2D/ReadVariableOpconv2d_5/Conv2D/ReadVariableOp2L
$latent_vector/BiasAdd/ReadVariableOp$latent_vector/BiasAdd/ReadVariableOp2J
#latent_vector/MatMul/ReadVariableOp#latent_vector/MatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
?
?
'__inference_decoder_layer_call_fn_88222
decoder_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldecoder_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+???????????????????????????*-
config_proto

GPU

CPU2*0J 8*K
fFRD
B__inference_decoder_layer_call_and_return_conditional_losses_882092
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*O
_input_shapes>
<:??????????::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:- )
'
_user_specified_namedecoder_input
?$
?
M__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_87976

inputs,
(conv2d_transpose_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/3?
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
2
conv2d_transpose?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:& "
 
_user_specified_nameinputs
?
?
'__inference_decoder_layer_call_fn_89225

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+???????????????????????????*-
config_proto

GPU

CPU2*0J 8*K
fFRD
B__inference_decoder_layer_call_and_return_conditional_losses_881742
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*O
_input_shapes>
<:??????????::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?
?
'__inference_dense_1_layer_call_fn_89285

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:?????????? *-
config_proto

GPU

CPU2*0J 8*K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_880842
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:?????????? 2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?
`
D__inference_reshape_1_layer_call_and_return_conditional_losses_89299

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2e
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :?2
Reshape/shape/3?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapex
ReshapeReshapeinputsReshape/shape:output:0*
T0*0
_output_shapes
:??????????2	
Reshapem
IdentityIdentityReshape:output:0*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:?????????? :& "
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
O
encoder_input>
serving_default_encoder_input:0?????????  C
decoder8
StatefulPartitionedCall:0?????????  tensorflow/serving/predict:??
?v
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
	optimizer
regularization_losses
	variables
trainable_variables
	keras_api
	
signatures
+?&call_and_return_all_conditional_losses
?_default_save_signature
?__call__"?s
_tf_keras_model?s{"class_name": "Model", "name": "autoencoder", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "autoencoder", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": [null, 32, 32, 1], "dtype": "float32", "sparse": false, "ragged": false, "name": "encoder_input"}, "name": "encoder_input", "inbound_nodes": []}, {"class_name": "Model", "config": {"name": "encoder", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": [null, 32, 32, 1], "dtype": "float32", "sparse": false, "ragged": false, "name": "encoder_input"}, "name": "encoder_input", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [3, 3], "strides": [2, 2], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_3", "inbound_nodes": [[["encoder_input", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_4", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": [3, 3], "strides": [2, 2], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_4", "inbound_nodes": [[["conv2d_3", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_5", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": [3, 3], "strides": [2, 2], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_5", "inbound_nodes": [[["conv2d_4", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_1", "inbound_nodes": [[["conv2d_5", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "latent_vector", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "latent_vector", "inbound_nodes": [[["flatten_1", 0, 0, {}]]]}], "input_layers": [["encoder_input", 0, 0]], "output_layers": [["latent_vector", 0, 0]]}, "name": "encoder", "inbound_nodes": [[["encoder_input", 0, 0, {}]]]}, {"class_name": "Model", "config": {"name": "decoder", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": [null, 256], "dtype": "float32", "sparse": false, "ragged": false, "name": "decoder_input"}, "name": "decoder_input", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 4096, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["decoder_input", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape_1", "trainable": true, "dtype": "float32", "target_shape": [4, 4, 256]}, "name": "reshape_1", "inbound_nodes": [[["dense_1", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_3", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": [3, 3], "strides": [2, 2], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv2d_transpose_3", "inbound_nodes": [[["reshape_1", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_4", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": [3, 3], "strides": [2, 2], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv2d_transpose_4", "inbound_nodes": [[["conv2d_transpose_3", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_5", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": [3, 3], "strides": [2, 2], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv2d_transpose_5", "inbound_nodes": [[["conv2d_transpose_4", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "decoder_output", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "decoder_output", "inbound_nodes": [[["conv2d_transpose_5", 0, 0, {}]]]}], "input_layers": [["decoder_input", 0, 0]], "output_layers": [["decoder_output", 0, 0]]}, "name": "decoder", "inbound_nodes": [[["encoder", 1, 0, {}]]]}], "input_layers": [["encoder_input", 0, 0]], "output_layers": [["decoder", 1, 0]]}, "is_graph_network": true, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Model", "config": {"name": "autoencoder", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": [null, 32, 32, 1], "dtype": "float32", "sparse": false, "ragged": false, "name": "encoder_input"}, "name": "encoder_input", "inbound_nodes": []}, {"class_name": "Model", "config": {"name": "encoder", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": [null, 32, 32, 1], "dtype": "float32", "sparse": false, "ragged": false, "name": "encoder_input"}, "name": "encoder_input", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [3, 3], "strides": [2, 2], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_3", "inbound_nodes": [[["encoder_input", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_4", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": [3, 3], "strides": [2, 2], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_4", "inbound_nodes": [[["conv2d_3", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_5", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": [3, 3], "strides": [2, 2], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_5", "inbound_nodes": [[["conv2d_4", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_1", "inbound_nodes": [[["conv2d_5", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "latent_vector", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "latent_vector", "inbound_nodes": [[["flatten_1", 0, 0, {}]]]}], "input_layers": [["encoder_input", 0, 0]], "output_layers": [["latent_vector", 0, 0]]}, "name": "encoder", "inbound_nodes": [[["encoder_input", 0, 0, {}]]]}, {"class_name": "Model", "config": {"name": "decoder", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": [null, 256], "dtype": "float32", "sparse": false, "ragged": false, "name": "decoder_input"}, "name": "decoder_input", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 4096, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["decoder_input", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape_1", "trainable": true, "dtype": "float32", "target_shape": [4, 4, 256]}, "name": "reshape_1", "inbound_nodes": [[["dense_1", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_3", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": [3, 3], "strides": [2, 2], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv2d_transpose_3", "inbound_nodes": [[["reshape_1", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_4", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": [3, 3], "strides": [2, 2], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv2d_transpose_4", "inbound_nodes": [[["conv2d_transpose_3", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_5", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": [3, 3], "strides": [2, 2], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv2d_transpose_5", "inbound_nodes": [[["conv2d_transpose_4", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "decoder_output", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "decoder_output", "inbound_nodes": [[["conv2d_transpose_5", 0, 0, {}]]]}], "input_layers": [["decoder_input", 0, 0]], "output_layers": [["decoder_output", 0, 0]]}, "name": "decoder", "inbound_nodes": [[["encoder", 1, 0, {}]]]}], "input_layers": [["encoder_input", 0, 0]], "output_layers": [["decoder", 1, 0]]}}, "training_config": {"loss": "mse", "metrics": [], "weighted_metrics": null, "sample_weight_mode": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0003162277862429619, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "encoder_input", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": [null, 32, 32, 1], "config": {"batch_input_shape": [null, 32, 32, 1], "dtype": "float32", "sparse": false, "ragged": false, "name": "encoder_input"}}
?1
layer-0

layer_with_weights-0

layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer-4
layer_with_weights-3
layer-5
regularization_losses
	variables
trainable_variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?/
_tf_keras_model?.{"class_name": "Model", "name": "encoder", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "encoder", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": [null, 32, 32, 1], "dtype": "float32", "sparse": false, "ragged": false, "name": "encoder_input"}, "name": "encoder_input", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [3, 3], "strides": [2, 2], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_3", "inbound_nodes": [[["encoder_input", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_4", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": [3, 3], "strides": [2, 2], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_4", "inbound_nodes": [[["conv2d_3", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_5", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": [3, 3], "strides": [2, 2], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_5", "inbound_nodes": [[["conv2d_4", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_1", "inbound_nodes": [[["conv2d_5", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "latent_vector", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "latent_vector", "inbound_nodes": [[["flatten_1", 0, 0, {}]]]}], "input_layers": [["encoder_input", 0, 0]], "output_layers": [["latent_vector", 0, 0]]}, "is_graph_network": true, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Model", "config": {"name": "encoder", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": [null, 32, 32, 1], "dtype": "float32", "sparse": false, "ragged": false, "name": "encoder_input"}, "name": "encoder_input", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [3, 3], "strides": [2, 2], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_3", "inbound_nodes": [[["encoder_input", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_4", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": [3, 3], "strides": [2, 2], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_4", "inbound_nodes": [[["conv2d_3", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_5", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": [3, 3], "strides": [2, 2], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_5", "inbound_nodes": [[["conv2d_4", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_1", "inbound_nodes": [[["conv2d_5", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "latent_vector", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "latent_vector", "inbound_nodes": [[["flatten_1", 0, 0, {}]]]}], "input_layers": [["encoder_input", 0, 0]], "output_layers": [["latent_vector", 0, 0]]}}}
?>
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer_with_weights-4
layer-6
regularization_losses
	variables
trainable_variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?<
_tf_keras_model?;{"class_name": "Model", "name": "decoder", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "decoder", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": [null, 256], "dtype": "float32", "sparse": false, "ragged": false, "name": "decoder_input"}, "name": "decoder_input", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 4096, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["decoder_input", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape_1", "trainable": true, "dtype": "float32", "target_shape": [4, 4, 256]}, "name": "reshape_1", "inbound_nodes": [[["dense_1", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_3", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": [3, 3], "strides": [2, 2], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv2d_transpose_3", "inbound_nodes": [[["reshape_1", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_4", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": [3, 3], "strides": [2, 2], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv2d_transpose_4", "inbound_nodes": [[["conv2d_transpose_3", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_5", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": [3, 3], "strides": [2, 2], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv2d_transpose_5", "inbound_nodes": [[["conv2d_transpose_4", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "decoder_output", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "decoder_output", "inbound_nodes": [[["conv2d_transpose_5", 0, 0, {}]]]}], "input_layers": [["decoder_input", 0, 0]], "output_layers": [["decoder_output", 0, 0]]}, "is_graph_network": true, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Model", "config": {"name": "decoder", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": [null, 256], "dtype": "float32", "sparse": false, "ragged": false, "name": "decoder_input"}, "name": "decoder_input", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 4096, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["decoder_input", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape_1", "trainable": true, "dtype": "float32", "target_shape": [4, 4, 256]}, "name": "reshape_1", "inbound_nodes": [[["dense_1", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_3", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": [3, 3], "strides": [2, 2], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv2d_transpose_3", "inbound_nodes": [[["reshape_1", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_4", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": [3, 3], "strides": [2, 2], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv2d_transpose_4", "inbound_nodes": [[["conv2d_transpose_3", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_5", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": [3, 3], "strides": [2, 2], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv2d_transpose_5", "inbound_nodes": [[["conv2d_transpose_4", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "decoder_output", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "decoder_output", "inbound_nodes": [[["conv2d_transpose_5", 0, 0, {}]]]}], "input_layers": [["decoder_input", 0, 0]], "output_layers": [["decoder_output", 0, 0]]}}}
?
iter

beta_1

 beta_2
	!decay
"learning_rate#m?$m?%m?&m?'m?(m?)m?*m?+m?,m?-m?.m?/m?0m?1m?2m?3m?4m?#v?$v?%v?&v?'v?(v?)v?*v?+v?,v?-v?.v?/v?0v?1v?2v?3v?4v?"
	optimizer
 "
trackable_list_wrapper
?
#0
$1
%2
&3
'4
(5
)6
*7
+8
,9
-10
.11
/12
013
114
215
316
417"
trackable_list_wrapper
?
#0
$1
%2
&3
'4
(5
)6
*7
+8
,9
-10
.11
/12
013
114
215
316
417"
trackable_list_wrapper
?
5metrics
regularization_losses
	variables
6layer_regularization_losses
7non_trainable_variables
trainable_variables

8layers
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
?

#kernel
$bias
9regularization_losses
:	variables
;trainable_variables
<	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [3, 3], "strides": [2, 2], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 1}}}}
?

%kernel
&bias
=regularization_losses
>	variables
?trainable_variables
@	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv2d_4", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": [3, 3], "strides": [2, 2], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}}}
?

'kernel
(bias
Aregularization_losses
B	variables
Ctrainable_variables
D	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv2d_5", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": [3, 3], "strides": [2, 2], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 128}}}}
?
Eregularization_losses
F	variables
Gtrainable_variables
H	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?

)kernel
*bias
Iregularization_losses
J	variables
Ktrainable_variables
L	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "latent_vector", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "latent_vector", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 4096}}}}
 "
trackable_list_wrapper
X
#0
$1
%2
&3
'4
(5
)6
*7"
trackable_list_wrapper
X
#0
$1
%2
&3
'4
(5
)6
*7"
trackable_list_wrapper
?
Mmetrics
regularization_losses
	variables
Nlayer_regularization_losses
Onon_trainable_variables
trainable_variables

Players
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "decoder_input", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": [null, 256], "config": {"batch_input_shape": [null, 256], "dtype": "float32", "sparse": false, "ragged": false, "name": "decoder_input"}}
?

+kernel
,bias
Qregularization_losses
R	variables
Strainable_variables
T	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 4096, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}}
?
Uregularization_losses
V	variables
Wtrainable_variables
X	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Reshape", "name": "reshape_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "reshape_1", "trainable": true, "dtype": "float32", "target_shape": [4, 4, 256]}}
?

-kernel
.bias
Yregularization_losses
Z	variables
[trainable_variables
\	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2DTranspose", "name": "conv2d_transpose_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv2d_transpose_3", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": [3, 3], "strides": [2, 2], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 256}}}}
?

/kernel
0bias
]regularization_losses
^	variables
_trainable_variables
`	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2DTranspose", "name": "conv2d_transpose_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv2d_transpose_4", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": [3, 3], "strides": [2, 2], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 3}}}}
?

1kernel
2bias
aregularization_losses
b	variables
ctrainable_variables
d	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2DTranspose", "name": "conv2d_transpose_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv2d_transpose_5", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": [3, 3], "strides": [2, 2], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 3}}}}
?

3kernel
4bias
eregularization_losses
f	variables
gtrainable_variables
h	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2DTranspose", "name": "decoder_output", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "decoder_output", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 3}}}}
 "
trackable_list_wrapper
f
+0
,1
-2
.3
/4
05
16
27
38
49"
trackable_list_wrapper
f
+0
,1
-2
.3
/4
05
16
27
38
49"
trackable_list_wrapper
?
imetrics
regularization_losses
	variables
jlayer_regularization_losses
knon_trainable_variables
trainable_variables

llayers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
):'@2conv2d_3/kernel
:@2conv2d_3/bias
*:(@?2conv2d_4/kernel
:?2conv2d_4/bias
+:)??2conv2d_5/kernel
:?2conv2d_5/bias
*:(
? ?2latent_vector_1/kernel
#:!?2latent_vector_1/bias
": 
?? 2dense_1/kernel
:? 2dense_1/bias
4:2?2conv2d_transpose_3/kernel
%:#2conv2d_transpose_3/bias
3:12conv2d_transpose_4/kernel
%:#2conv2d_transpose_4/bias
3:12conv2d_transpose_5/kernel
%:#2conv2d_transpose_5/bias
1:/2decoder_output_1/kernel
#:!2decoder_output_1/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
?
mmetrics
9regularization_losses
:	variables
nlayer_regularization_losses
onon_trainable_variables
;trainable_variables

players
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
?
qmetrics
=regularization_losses
>	variables
rlayer_regularization_losses
snon_trainable_variables
?trainable_variables

tlayers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
?
umetrics
Aregularization_losses
B	variables
vlayer_regularization_losses
wnon_trainable_variables
Ctrainable_variables

xlayers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
ymetrics
Eregularization_losses
F	variables
zlayer_regularization_losses
{non_trainable_variables
Gtrainable_variables

|layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
?
}metrics
Iregularization_losses
J	variables
~layer_regularization_losses
non_trainable_variables
Ktrainable_variables
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
J
0

1
2
3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
?
?metrics
Qregularization_losses
R	variables
 ?layer_regularization_losses
?non_trainable_variables
Strainable_variables
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?metrics
Uregularization_losses
V	variables
 ?layer_regularization_losses
?non_trainable_variables
Wtrainable_variables
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
?
?metrics
Yregularization_losses
Z	variables
 ?layer_regularization_losses
?non_trainable_variables
[trainable_variables
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
/0
01"
trackable_list_wrapper
.
/0
01"
trackable_list_wrapper
?
?metrics
]regularization_losses
^	variables
 ?layer_regularization_losses
?non_trainable_variables
_trainable_variables
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
?
?metrics
aregularization_losses
b	variables
 ?layer_regularization_losses
?non_trainable_variables
ctrainable_variables
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
?
?metrics
eregularization_losses
f	variables
 ?layer_regularization_losses
?non_trainable_variables
gtrainable_variables
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Q
0
1
2
3
4
5
6"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.:,@2Adam/conv2d_3/kernel/m
 :@2Adam/conv2d_3/bias/m
/:-@?2Adam/conv2d_4/kernel/m
!:?2Adam/conv2d_4/bias/m
0:.??2Adam/conv2d_5/kernel/m
!:?2Adam/conv2d_5/bias/m
/:-
? ?2Adam/latent_vector_1/kernel/m
(:&?2Adam/latent_vector_1/bias/m
':%
?? 2Adam/dense_1/kernel/m
 :? 2Adam/dense_1/bias/m
9:7?2 Adam/conv2d_transpose_3/kernel/m
*:(2Adam/conv2d_transpose_3/bias/m
8:62 Adam/conv2d_transpose_4/kernel/m
*:(2Adam/conv2d_transpose_4/bias/m
8:62 Adam/conv2d_transpose_5/kernel/m
*:(2Adam/conv2d_transpose_5/bias/m
6:42Adam/decoder_output_1/kernel/m
(:&2Adam/decoder_output_1/bias/m
.:,@2Adam/conv2d_3/kernel/v
 :@2Adam/conv2d_3/bias/v
/:-@?2Adam/conv2d_4/kernel/v
!:?2Adam/conv2d_4/bias/v
0:.??2Adam/conv2d_5/kernel/v
!:?2Adam/conv2d_5/bias/v
/:-
? ?2Adam/latent_vector_1/kernel/v
(:&?2Adam/latent_vector_1/bias/v
':%
?? 2Adam/dense_1/kernel/v
 :? 2Adam/dense_1/bias/v
9:7?2 Adam/conv2d_transpose_3/kernel/v
*:(2Adam/conv2d_transpose_3/bias/v
8:62 Adam/conv2d_transpose_4/kernel/v
*:(2Adam/conv2d_transpose_4/bias/v
8:62 Adam/conv2d_transpose_5/kernel/v
*:(2Adam/conv2d_transpose_5/bias/v
6:42Adam/decoder_output_1/kernel/v
(:&2Adam/decoder_output_1/bias/v
?2?
F__inference_autoencoder_layer_call_and_return_conditional_losses_88303
F__inference_autoencoder_layer_call_and_return_conditional_losses_88623
F__inference_autoencoder_layer_call_and_return_conditional_losses_88792
F__inference_autoencoder_layer_call_and_return_conditional_losses_88327?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
 __inference__wrapped_model_87707?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *4?1
/?,
encoder_input?????????  
?2?
+__inference_autoencoder_layer_call_fn_88815
+__inference_autoencoder_layer_call_fn_88422
+__inference_autoencoder_layer_call_fn_88838
+__inference_autoencoder_layer_call_fn_88375?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
B__inference_encoder_layer_call_and_return_conditional_losses_87837
B__inference_encoder_layer_call_and_return_conditional_losses_88904
B__inference_encoder_layer_call_and_return_conditional_losses_87820
B__inference_encoder_layer_call_and_return_conditional_losses_88871?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
'__inference_encoder_layer_call_fn_88930
'__inference_encoder_layer_call_fn_88917
'__inference_encoder_layer_call_fn_87868
'__inference_encoder_layer_call_fn_87898?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
B__inference_decoder_layer_call_and_return_conditional_losses_88131
B__inference_decoder_layer_call_and_return_conditional_losses_89070
B__inference_decoder_layer_call_and_return_conditional_losses_88151
B__inference_decoder_layer_call_and_return_conditional_losses_89210?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
'__inference_decoder_layer_call_fn_88222
'__inference_decoder_layer_call_fn_89240
'__inference_decoder_layer_call_fn_89225
'__inference_decoder_layer_call_fn_88187?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
8B6
#__inference_signature_wrapper_88454encoder_input
?2?
C__inference_conv2d_3_layer_call_and_return_conditional_losses_87720?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+???????????????????????????
?2?
(__inference_conv2d_3_layer_call_fn_87728?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+???????????????????????????
?2?
C__inference_conv2d_4_layer_call_and_return_conditional_losses_87741?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+???????????????????????????@
?2?
(__inference_conv2d_4_layer_call_fn_87749?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+???????????????????????????@
?2?
C__inference_conv2d_5_layer_call_and_return_conditional_losses_87762?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *8?5
3?0,????????????????????????????
?2?
(__inference_conv2d_5_layer_call_fn_87770?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *8?5
3?0,????????????????????????????
?2?
D__inference_flatten_1_layer_call_and_return_conditional_losses_89246?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_flatten_1_layer_call_fn_89251?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_latent_vector_layer_call_and_return_conditional_losses_89261?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_latent_vector_layer_call_fn_89268?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_dense_1_layer_call_and_return_conditional_losses_89278?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_dense_1_layer_call_fn_89285?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_reshape_1_layer_call_and_return_conditional_losses_89299?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_reshape_1_layer_call_fn_89304?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
M__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_87933?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *8?5
3?0,????????????????????????????
?2?
2__inference_conv2d_transpose_3_layer_call_fn_87941?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *8?5
3?0,????????????????????????????
?2?
M__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_87976?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+???????????????????????????
?2?
2__inference_conv2d_transpose_4_layer_call_fn_87984?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+???????????????????????????
?2?
M__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_88019?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+???????????????????????????
?2?
2__inference_conv2d_transpose_5_layer_call_fn_88027?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+???????????????????????????
?2?
I__inference_decoder_output_layer_call_and_return_conditional_losses_88062?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+???????????????????????????
?2?
.__inference_decoder_output_layer_call_fn_88070?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+????????????????????????????
 __inference__wrapped_model_87707?#$%&'()*+,-./01234>?;
4?1
/?,
encoder_input?????????  
? "9?6
4
decoder)?&
decoder?????????  ?
F__inference_autoencoder_layer_call_and_return_conditional_losses_88303?#$%&'()*+,-./01234F?C
<?9
/?,
encoder_input?????????  
p

 
? "??<
5?2
0+???????????????????????????
? ?
F__inference_autoencoder_layer_call_and_return_conditional_losses_88327?#$%&'()*+,-./01234F?C
<?9
/?,
encoder_input?????????  
p 

 
? "??<
5?2
0+???????????????????????????
? ?
F__inference_autoencoder_layer_call_and_return_conditional_losses_88623?#$%&'()*+,-./01234??<
5?2
(?%
inputs?????????  
p

 
? "-?*
#? 
0?????????  
? ?
F__inference_autoencoder_layer_call_and_return_conditional_losses_88792?#$%&'()*+,-./01234??<
5?2
(?%
inputs?????????  
p 

 
? "-?*
#? 
0?????????  
? ?
+__inference_autoencoder_layer_call_fn_88375?#$%&'()*+,-./01234F?C
<?9
/?,
encoder_input?????????  
p

 
? "2?/+????????????????????????????
+__inference_autoencoder_layer_call_fn_88422?#$%&'()*+,-./01234F?C
<?9
/?,
encoder_input?????????  
p 

 
? "2?/+????????????????????????????
+__inference_autoencoder_layer_call_fn_88815?#$%&'()*+,-./01234??<
5?2
(?%
inputs?????????  
p

 
? "2?/+????????????????????????????
+__inference_autoencoder_layer_call_fn_88838?#$%&'()*+,-./01234??<
5?2
(?%
inputs?????????  
p 

 
? "2?/+????????????????????????????
C__inference_conv2d_3_layer_call_and_return_conditional_losses_87720?#$I?F
??<
:?7
inputs+???????????????????????????
? "??<
5?2
0+???????????????????????????@
? ?
(__inference_conv2d_3_layer_call_fn_87728?#$I?F
??<
:?7
inputs+???????????????????????????
? "2?/+???????????????????????????@?
C__inference_conv2d_4_layer_call_and_return_conditional_losses_87741?%&I?F
??<
:?7
inputs+???????????????????????????@
? "@?=
6?3
0,????????????????????????????
? ?
(__inference_conv2d_4_layer_call_fn_87749?%&I?F
??<
:?7
inputs+???????????????????????????@
? "3?0,?????????????????????????????
C__inference_conv2d_5_layer_call_and_return_conditional_losses_87762?'(J?G
@?=
;?8
inputs,????????????????????????????
? "@?=
6?3
0,????????????????????????????
? ?
(__inference_conv2d_5_layer_call_fn_87770?'(J?G
@?=
;?8
inputs,????????????????????????????
? "3?0,?????????????????????????????
M__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_87933?-.J?G
@?=
;?8
inputs,????????????????????????????
? "??<
5?2
0+???????????????????????????
? ?
2__inference_conv2d_transpose_3_layer_call_fn_87941?-.J?G
@?=
;?8
inputs,????????????????????????????
? "2?/+????????????????????????????
M__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_87976?/0I?F
??<
:?7
inputs+???????????????????????????
? "??<
5?2
0+???????????????????????????
? ?
2__inference_conv2d_transpose_4_layer_call_fn_87984?/0I?F
??<
:?7
inputs+???????????????????????????
? "2?/+????????????????????????????
M__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_88019?12I?F
??<
:?7
inputs+???????????????????????????
? "??<
5?2
0+???????????????????????????
? ?
2__inference_conv2d_transpose_5_layer_call_fn_88027?12I?F
??<
:?7
inputs+???????????????????????????
? "2?/+????????????????????????????
B__inference_decoder_layer_call_and_return_conditional_losses_88131?
+,-./01234??<
5?2
(?%
decoder_input??????????
p

 
? "??<
5?2
0+???????????????????????????
? ?
B__inference_decoder_layer_call_and_return_conditional_losses_88151?
+,-./01234??<
5?2
(?%
decoder_input??????????
p 

 
? "??<
5?2
0+???????????????????????????
? ?
B__inference_decoder_layer_call_and_return_conditional_losses_89070u
+,-./012348?5
.?+
!?
inputs??????????
p

 
? "-?*
#? 
0?????????  
? ?
B__inference_decoder_layer_call_and_return_conditional_losses_89210u
+,-./012348?5
.?+
!?
inputs??????????
p 

 
? "-?*
#? 
0?????????  
? ?
'__inference_decoder_layer_call_fn_88187?
+,-./01234??<
5?2
(?%
decoder_input??????????
p

 
? "2?/+????????????????????????????
'__inference_decoder_layer_call_fn_88222?
+,-./01234??<
5?2
(?%
decoder_input??????????
p 

 
? "2?/+????????????????????????????
'__inference_decoder_layer_call_fn_89225z
+,-./012348?5
.?+
!?
inputs??????????
p

 
? "2?/+????????????????????????????
'__inference_decoder_layer_call_fn_89240z
+,-./012348?5
.?+
!?
inputs??????????
p 

 
? "2?/+????????????????????????????
I__inference_decoder_output_layer_call_and_return_conditional_losses_88062?34I?F
??<
:?7
inputs+???????????????????????????
? "??<
5?2
0+???????????????????????????
? ?
.__inference_decoder_output_layer_call_fn_88070?34I?F
??<
:?7
inputs+???????????????????????????
? "2?/+????????????????????????????
B__inference_dense_1_layer_call_and_return_conditional_losses_89278^+,0?-
&?#
!?
inputs??????????
? "&?#
?
0?????????? 
? |
'__inference_dense_1_layer_call_fn_89285Q+,0?-
&?#
!?
inputs??????????
? "??????????? ?
B__inference_encoder_layer_call_and_return_conditional_losses_87820z#$%&'()*F?C
<?9
/?,
encoder_input?????????  
p

 
? "&?#
?
0??????????
? ?
B__inference_encoder_layer_call_and_return_conditional_losses_87837z#$%&'()*F?C
<?9
/?,
encoder_input?????????  
p 

 
? "&?#
?
0??????????
? ?
B__inference_encoder_layer_call_and_return_conditional_losses_88871s#$%&'()*??<
5?2
(?%
inputs?????????  
p

 
? "&?#
?
0??????????
? ?
B__inference_encoder_layer_call_and_return_conditional_losses_88904s#$%&'()*??<
5?2
(?%
inputs?????????  
p 

 
? "&?#
?
0??????????
? ?
'__inference_encoder_layer_call_fn_87868m#$%&'()*F?C
<?9
/?,
encoder_input?????????  
p

 
? "????????????
'__inference_encoder_layer_call_fn_87898m#$%&'()*F?C
<?9
/?,
encoder_input?????????  
p 

 
? "????????????
'__inference_encoder_layer_call_fn_88917f#$%&'()*??<
5?2
(?%
inputs?????????  
p

 
? "????????????
'__inference_encoder_layer_call_fn_88930f#$%&'()*??<
5?2
(?%
inputs?????????  
p 

 
? "????????????
D__inference_flatten_1_layer_call_and_return_conditional_losses_89246b8?5
.?+
)?&
inputs??????????
? "&?#
?
0?????????? 
? ?
)__inference_flatten_1_layer_call_fn_89251U8?5
.?+
)?&
inputs??????????
? "??????????? ?
H__inference_latent_vector_layer_call_and_return_conditional_losses_89261^)*0?-
&?#
!?
inputs?????????? 
? "&?#
?
0??????????
? ?
-__inference_latent_vector_layer_call_fn_89268Q)*0?-
&?#
!?
inputs?????????? 
? "????????????
D__inference_reshape_1_layer_call_and_return_conditional_losses_89299b0?-
&?#
!?
inputs?????????? 
? ".?+
$?!
0??????????
? ?
)__inference_reshape_1_layer_call_fn_89304U0?-
&?#
!?
inputs?????????? 
? "!????????????
#__inference_signature_wrapper_88454?#$%&'()*+,-./01234O?L
? 
E?B
@
encoder_input/?,
encoder_input?????????  "9?6
4
decoder)?&
decoder?????????  