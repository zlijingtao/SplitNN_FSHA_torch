import torch
import numpy as np
import torch.nn as nn
import architectures_torch as architectures
import tqdm

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight, gain=1.0)
        m.bias.data.zero_()
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        torch.nn.init.xavier_uniform_(m.weight, gain=1.0)
        m.bias.data.zero_()

def distance_data_loss(a,b):
    l = nn.MSELoss()
    return l(a, b)

def distance_data(a,b):
    l = nn.MSELoss()
    return l(a, b)

def zeroing_grad(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            param.grad = torch.zeros_like(param.grad).to(param.device)

class FSHA:
    
    def loadBiasNetwork(self, make_decoder, z_shape, channels):
        return make_decoder(z_shape, channels=channels)
        
    def __init__(self, xpriv, xpub, id_setup, batch_size, hparams):
            input_shape = xpriv[0][0].shape

            self.hparams = hparams

            # setup dataset
            self.client_dataset = torch.utils.data.DataLoader(xpriv,  batch_size=batch_size, shuffle=True,
                num_workers=4, pin_memory=True)
            self.attacker_dataset = torch.utils.data.DataLoader(xpub,  batch_size=batch_size, shuffle=True,
                num_workers=4, pin_memory=True)

            self.batch_size = batch_size

            ## setup models
            make_f, make_tilde_f, make_decoder, make_D = architectures.SETUPS[id_setup]

            self.f = make_f(input_shape)
            self.tilde_f = make_tilde_f(input_shape)

            test_input = torch.zeros([1, input_shape[0], input_shape[1], input_shape[2]])
            f_out = self.f(test_input)
            tilde_f_out = self.tilde_f(test_input)
            assert f_out.size()[1:] == tilde_f_out.size()[1:]
            z_shape = tilde_f_out.size()[1:]
            print(z_shape)
            self.D = make_D(z_shape)
            self.decoder = self.loadBiasNetwork(make_decoder, z_shape, channels=input_shape[0])
            
            #initialize modules
            self.f.apply(init_weights)
            self.tilde_f.apply(init_weights)
            self.D.apply(init_weights)
            self.decoder.apply(init_weights)

            # move models to GPU
            self.f.cuda()
            self.tilde_f.cuda()
            self.D.cuda()
            self.decoder.cuda()

            # setup optimizers
            self.optimizer0 = torch.optim.Adam(self.f.parameters(), lr=hparams['lr_f'])
            self.optimizer1 = torch.optim.Adam([{'params': self.tilde_f.parameters()}, {'params': self.decoder.parameters()}], lr=hparams['lr_tilde'])
            self.optimizer2 = torch.optim.Adam(self.D.parameters(), lr=hparams['lr_D'])


    @staticmethod
    def addNoise(x, alpha):
        return x + torch.randn(x.size()) * alpha

    def train_step(self, x_private, x_public, label_private, label_public):
        torch.autograd.set_detect_anomaly(True)
        self.f.train()
        self.tilde_f.train()
        self.decoder.train()
        self.D.train()

        x_private = x_private.cuda(non_blocking=False)
        x_public = x_public.cuda(non_blocking=False)

        #### Virtually, ON THE CLIENT SIDE:
        # clients' smashed data
        z_private = self.f(x_private)
        ####################################

        #### SERVER-SIDE:
        
        ## adversarial loss (f's output must similar be to \tilde{f}'s output):
        adv_private_logits = self.D(z_private)
        
        if self.hparams['WGAN']:
            # print("Use WGAN loss")
            f_loss = torch.mean(adv_private_logits)
        else:
            criterion = torch.nn.BCELoss()
            f_loss = criterion(adv_private_logits, torch.ones_like(adv_private_logits.detach()))
        ##

        z_public = self.tilde_f(x_public)

        # invertibility loss
        rec_x_public = self.decoder(z_public)
        public_rec_loss = distance_data_loss(x_public, rec_x_public)
        tilde_f_loss = public_rec_loss

        # discriminator on attacker's feature-space
        adv_public_logits = self.D(z_public.detach())
        adv_private_logits_detached = self.D(z_private.detach())

        if self.hparams['WGAN']:
            loss_discr_true = torch.mean( adv_public_logits )
            loss_discr_fake = -torch.mean( adv_private_logits_detached)
            # discriminator's loss
            D_loss = loss_discr_true + loss_discr_fake
        else:
            criterion = nn.BCELoss()
            loss_discr_true = criterion(adv_public_logits, torch.ones_like(adv_public_logits.detach()))
            loss_discr_fake = criterion(adv_private_logits_detached, torch.zeros_like(adv_private_logits_detached.detach()))
            # discriminator's loss
            D_loss = (loss_discr_true + loss_discr_fake) / 2

        if 'gradient_penalty' in self.hparams:
            # print("Use GP")
            w = float(self.hparams['gradient_penalty'])
            D_gradient_penalty = self.gradient_penalty(z_private.detach(), z_public.detach())
            D_loss += D_gradient_penalty * w

        ##################################################################
        ## attack validation #####################
        with torch.no_grad():
            # map to data space (for evaluation and style loss)
            rec_x_private = self.decoder(z_private)
            loss_c_verification = distance_data(x_private, rec_x_private)
            losses_c_verification = loss_c_verification.detach()
            del rec_x_private, loss_c_verification

        self.optimizer0.zero_grad()
        self.optimizer1.zero_grad()
        self.optimizer2.zero_grad()

        # train client's network 
        f_loss.backward()
        
        zeroing_grad(self.D)

        # train attacker's autoencoder on public data
        tilde_f_loss.backward()
        
        # train discriminator
        D_loss.backward()

        self.optimizer0.step()
        
        self.optimizer1.step()
        
        self.optimizer2.step()

        f_losses = f_loss.detach()
        tilde_f_losses = tilde_f_loss.detach()
        D_losses = D_loss.detach()
        

        del f_loss, tilde_f_loss, D_loss

        return f_losses, tilde_f_losses, D_losses, losses_c_verification


    def gradient_penalty(self, x, x_gen):
        epsilon = torch.rand([x.shape[0], 1, 1, 1]).cuda()
        x_hat = epsilon * x + (1 - epsilon) * x_gen
        x_hat = torch.autograd.Variable(x_hat, requires_grad=True)
        from torch.autograd import grad
        d_hat = self.D(x_hat)
        gradients = grad(outputs=d_hat, inputs=x_hat,
                        grad_outputs=torch.ones_like(d_hat).cuda(),
                        retain_graph=True, create_graph=True, only_inputs=True)[0]
        gradients = gradients.view(gradients.size(0),  -1)
        gradient_norm = gradients.norm(2, dim=1)
        penalty = ((gradient_norm - 1)**2).mean()
        return penalty

    def attack(self, x_private):
        with torch.no_grad():
            # smashed data sent from the client:
            z_private = self.f(x_private)
            # recover private data from smashed data
            tilde_x_private = self.decoder(z_private)

            z_private_control = self.tilde_f(x_private)
            control = self.decoder(z_private_control)
        return tilde_x_private, control


    def __call__(self, iterations, log_frequency=500, verbose=False, progress_bar=True):

        n = int(iterations / log_frequency)
        LOG = np.zeros((n, 4))

        client_iterator = iter(self.client_dataset)
        attacker_iterator = iter(self.attacker_dataset)
        print("RUNNING...")
        iterator = list(range(iterations))
        j = 0

        for i in tqdm.tqdm(iterator , total=iterations):
            try:
                x_private, label_private = next(client_iterator)
                if x_private.size(0) != self.batch_size:
                    client_iterator = iter(self.client_dataset)
                    x_private, label_private = next(client_iterator)
            except StopIteration:
                client_iterator = iter(self.client_dataset)
                x_private, label_private = next(client_iterator)
            try:
                x_public, label_public = next(attacker_iterator)
                if x_public.size(0) != self.batch_size:
                    attacker_iterator = iter(self.attacker_dataset)
                    x_public, label_public = next(attacker_iterator)
            except StopIteration:
                attacker_iterator = iter(self.attacker_dataset)
                x_public, label_public = next(attacker_iterator)
            log = self.train_step(x_private, x_public, label_private, label_public)

            if i == 0:
                VAL = log[3]
            else:
                VAL += log[3] / log_frequency

            if  i % log_frequency == 0:
                LOG[j] = log

                if verbose:
                    print("log--%02d%%-%07d] validation: %0.4f" % ( int(i/iterations*100) ,i, VAL) )
                    print("f_Loss: {}\nf_tilde_loss: {}\nD_loss: {}\n".format(log[0], log[1], log[2]))
                VAL = 0
                j += 1
            
        return LOG

#----------------------------------------------------------------------------------------------------------------------


class FSHA_binary_property(FSHA):
    
    def loadBiasNetwork(self, make_decoder, z_shape, channels):
        class_num = self.hparams.get("class_num", 1)
        return make_decoder(z_shape, class_num)
    
    def binary_accuracy(self, label, logits):
    
        if self.hparams.get('class_num', 1) == 1:
            p = nn.Sigmoid(logits)
            predicted = torch.cast( (p > .5), torch.float32)
        else:
            p = nn.Softmax(logits)
            predicted = torch.argmax(p, 1)

        correct_prediction = torch.equal(label, predicted)
        return torch.mean(torch.cast(correct_prediction, torch.float32))

    
    def classification_loss(self, label, logits):
        
        if self.hparams.get('class_num', 1) == 1:
            criterion = nn.BCELoss()
        else:
            criterion = nn.CrossEntropyLoss()
        return criterion(logits, label)

    def train_step(self, x_private, x_public, label_private, label_public):

        torch.autograd.set_detect_anomaly(True)
        self.f.train()
        self.tilde_f.train()
        self.decoder.train()
        self.D.train()

        x_private = x_private.cuda(non_blocking=False)
        x_public = x_public.cuda(non_blocking=False)

        #### Virtually, ON THE CLIENT SIDE:
        # clients' smashed data
        z_private = self.f(x_private)
        ####################################

        #### SERVER-SIDE:
        # map to data space (for evaluation and style loss)
        clss_private_logits = self.decoder(z_private)
        ## adversarial loss (f's output must similar be to \tilde{f}'s output):
        adv_private_logits = self.D(z_private)
        
        if self.hparams['WGAN']:
            # print("Use WGAN loss")
            f_loss = torch.mean(adv_private_logits)
        else:
            criterion = torch.nn.BCELoss()
            f_loss = criterion(adv_private_logits, torch.ones_like(adv_private_logits.detach()))

        # attacker's classifier
        z_public = self.tilde_f(x_public)
        clss_public_logits = self.decoder(z_public)

        # classificatio loss
        public_classification_loss = self.classification_loss(label_public, clss_public_logits)
        
        tilde_f_loss = public_classification_loss

        adv_public_logits = self.D(z_public.detach())
        adv_private_logits_detached = self.D(z_private.detach())

        if self.hparams['WGAN']:
            loss_discr_true = torch.mean( adv_public_logits )
            loss_discr_fake = -torch.mean( adv_private_logits_detached)
            # discriminator's loss
            D_loss = loss_discr_true + loss_discr_fake
        else:
            criterion = nn.BCELoss()
            loss_discr_true = criterion(adv_public_logits, torch.ones_like(adv_public_logits.detach()))
            loss_discr_fake = criterion(adv_private_logits_detached, torch.zeros_like(adv_private_logits_detached.detach()))
            # discriminator's loss
            D_loss = (loss_discr_true + loss_discr_fake) / 2

        if 'gradient_penalty' in self.hparams:
            # print("Use GP")
            w = float(self.hparams['gradient_penalty'])
            D_gradient_penalty = self.gradient_penalty(z_private.detach(), z_public.detach())
            D_loss += D_gradient_penalty * w

        ##################################################################
        ## attack validation #####################
        with torch.no_grad():
            public_classification_accuracy = self.binary_accuracy(label_public, clss_public_logits)
            private_classification_accuracy = self.binary_accuracy(label_private, clss_private_logits)
            private_classification_accuracy_detached = private_classification_accuracy.detach()
            public_classification_accuracy_detached = public_classification_accuracy.detach()
            del private_classification_accuracy, public_classification_accuracy
        ############################################
        ##################################################################

        self.optimizer0.zero_grad()
        self.optimizer1.zero_grad()
        self.optimizer2.zero_grad()

        # train client's network 
        f_loss.backward()
        
        zeroing_grad(self.D)

        # train attacker's autoencoder on public data
        tilde_f_loss.backward()
        
        # train discriminator
        D_loss.backward()

        self.optimizer0.step()
        
        self.optimizer1.step()
        
        self.optimizer2.step()

        f_losses = f_loss.detach()
        tilde_f_losses = tilde_f_loss.detach()
        D_losses = D_loss.detach()
        
        del f_loss, tilde_f_loss, D_loss

        return f_losses, tilde_f_losses, D_losses, private_classification_accuracy_detached, public_classification_accuracy_detached