# version-control  
使用GCL方法的流程 (https://github.com/chenhao2345/GCL)  
讀取參數、影像、3D pose以及前處理  
使用GCL的模型架構，使用其 E_c (content encoder), E_p (pose encoder), D, G, mlp and memory  
GANLoss使用MUNIT的算法結合PDA-GAN的smooth label and noise to D  
G loss有使用GCL的feature recon. loss  
  
## **使用兩組G和D:**  
[G, mlp, Dt] and [F, mlp, Ds]  

## **使用optimizers:**  
opti_G: [G, mlp, E_p] with Adam(lr=0.0001, beta=[0.5, 0.999], weight_decay=0.0001)  
opti_F: [F, mlp, E_p] with Adam(lr=0.0001, beta=[0.5, 0.999], weight_decay=0.0001)  
opti_D: [D, Dt] with Adam(lr=0.0001, beta=[0.5, 0.999], weight_decay=0.0001)  
opti_E_c: [E_c] with SGD(lr=0.00035, weight_decay=0.0001, momentum=0.9, nesterov=True)  
update lr 10次(step size = iter./10)  
E_c scheduler gamma = 0.1  

## **使用WGAN-GP的設計**  
1. 使用smooth label and noise to D  
2. backward 完 D loss 之後，在 backward gradient_penalty (util.calc_gradient_penalty)  

## **使用GCL的encode and decode**  
image = [batch, 3, 256, 128]  
pose = [batch, 1, 256, 128]  
content = E_c(image) = [batch, 8192]  
pose_f = E_p(pose) = [batch, 128, 64, 32]  
pred = D(image)  
pred: list, len = 3  
[batch, 1, 64, 32]  
[batch, 1, 32, 16]  
[batch, 1, 16, 8]  

## **Ds loss:**  
1. loss_Ds_real = GANLoss(Ds(xs), True)  
2. loss_Ds_fake = GANLoss(Ds(x_t2s), False)  
3. gradient_penalty_loss  

## **Dt loss:**  
1. loss_Dt_real = GANLoss(Dt(xt), True)  
2. loss_Dt_fake = GANLoss(Dt(x_s2t), False)  
3. gradient_penalty_loss  

## **G loss:**  
1. loss_gen_adv_Dt = GANLoss(Dt(x_s2t), True)  
2. loss_recon_t2t = L1(x_t2t, xt)  
3. loss_cycrecon_t2s2t = L1(x_t2s2t, xt)  
4. loss_content_s2t = L1(E_c(x_s2t), E_c(xs))  
5. loss_content_t2s2t = L1(E_c(x_t2s2t), E_c(xt))   
6. loss_memory = memory(f_s, f_x_s2t) if stage == 3  

## **F loss:**  
1. loss_gen_adv_Ds = GANLoss(Ds(x_t2s), True)  
2. loss_recon_s2s = L1(x_s2s, xs)  
3. loss_cycrecon_s2t2s = L1(x_s2t2s, xs)  
4. loss_content_t2s = L1(E_c(x_t2s), E_c(xt))  
5. loss_content_s2t2s = L1(E_c(x_s2t2s), E_c(xs))  

## **Backward:**  
Ds.backward  
Dt.backward  
opti_D.step  
G.backward    
opti_G.step  
opti_F.step  
opti_Ec.step if stage == 3  
check update_learning_rate  

## ** Process:**
stage 2:  
save best loss G model  
stage 3:  
load stage 2 checkpoints  
init. memory  
compute mAP at snapshot iter  
save best mAP model  

## **example:**  
	checkpoint: 存model weight  
	data: 按照GCL放data  
	mesh: 按照GCL放mesh  
	output: 存生成圖片  
	training.py: 執行這個跑訓練  
## **model:**  
	losses.py: loss function  
	networks: define model  
	trainer: 整個訓練流程  
	其他: 幾乎參考GCL  
	
## **Notes:**  
1. model/datasets中可以更改dataset寫法，來讀取指定的camera id, dukemtmc.py in line 100, market1501.py in line 80.  
## **To do:**  
1. encoder需要獨立的opti嗎  
2. 目前content feature and pose feature 都需要在每個backward裡面都計算一次，試試看能不能計算一次之後給全部backward使用  
3. G and D adv loss都無法下降，可以分開顯示loss_D_real and loss_D__fake看看是哪個降不下來  
4. GANLoss中有一個 data = F.sigmoid(pred)，不確定是否會影響結果  
5. 想domain encoder怎麼和對比學習結合  
6. 需要warm up E_c and E_d嗎
7. MUNIT Adam有用weight_decay = 0.0001, 可以測試一下這樣會不會比較好  
	