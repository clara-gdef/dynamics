import pytorch_lightning as pl
import torch
import ipdb


class Disentangler(pl.LightningModule):
    def __init__(self, data_dir, desc, hparams):
        super().__init__()
        self.description = desc
        self.data_dir = data_dir

        # models
        self.domain_encoder = AtnWordEncoder()
        self.experience_encoder = AtnWordEncoder()
        self.industry_classifier = Classifier()
        self.desc_decoder = AtnWordDecoder()
        self.title_decoder = AtnWordDecoder()

        # variables
        self.domain_reps = None

        # hyperparams
        self.coeff = [1.]*6

    def forward(self, job):
        domain_job_rep = self.domain_encoder.forward(job)
        experience_job_rep = self.experience_encoder.forward(job)
        predicted_domain_ind = self.industry_classifier.forward(domain_job_rep)
        predicted_desc = self.desc_decoder.forward(domain_job_rep)
        predicted_exp_ind = self.industry_classifier.forward(experience_job_rep)
        predicted_title = self.title_decoder(experience_job_rep)

        return domain_job_rep, experience_job_rep, predicted_domain_ind, predicted_desc, predicted_exp_ind, predicted_title

    def on_train_epoch_start(self):
        ipdb.set_trace()
        # build domain reps

    def training_step(self, mini_batch, batch_nb):
        ipdb.set_trace()
        id_p, jobs_emb, jobs_titles, jobs_descriptions, industry = mini_batch
        # all element of batch, until before last one, on all dimensions
        loss_neg_sampling, loss_classif_domain, loss_desc, loss_title, loss_rk, loss_classif_exp = 0, 0, 0, 0, 0, 0
        for num, job_emb in enumerate(jobs_emb[:, :-1, :]):
            domain_job_rep, experience_job_rep, predicted_domain_ind, predicted_desc, predicted_exp_ind, predicted_title = self.forward(job_emb, num)
            # domain loss
            loss_neg_sampling += torch.nn.functional.triplet_margin_with_distance_loss(domain_job_rep,
                                                                                      self.domain_reps[industry],
                                                                                      self.sample_neg_examples(industry, self.hp.neg_examples))
            loss_classif_domain += torch.nn.functional.cross_entropy(predicted_domain_ind, industry)
            loss_desc += torch.nn.functional.cross_entropy(predicted_desc, jobs_descriptions[:, num, :])

            # experience loss
            prev_job_rep = self.experience_encoder.forward(jobs_emb[:, num + 1, :])
            loss_rk += torch.nn.functional.margin_ranking_loss(experience_job_rep, prev_job_rep,
                                                              torch.ones(self.hp.b_size))
            loss_classif_exp += - torch.nn.functional.cross_entropy(predicted_exp_ind, industry)
            loss_title += torch.nn.functional.cross_entropy(predicted_title, jobs_titles[:, num, :])


        domain_loss = (self.coeff[0] * loss_neg_sampling / num +
                       self.coeff[1] * loss_classif_domain / num +
                       self.coeff[2] * loss_desc / num) / sum(self.coeff[:3])
        experience_loss = (self.coeff[3] * loss_rk / num +
                           self.coeff[4] * loss_classif_exp  / num+
                           self.coeff[5] * loss_title / num) / sum(self.coeff[3:])

        self.log_losses_train(loss_neg_sampling, loss_classif_domain, loss_desc, loss_rk, loss_classif_exp, loss_title, experience_loss, domain_loss)
        return {"train_exp_loss": experience_loss, "train_domain_loss": domain_loss}


    def validation_step(self, mini_batch, batch_nb):
        ipdb.set_trace()
        id_p, jobs_emb, jobs_titles, jobs_descriptions, industry = mini_batch
        # all element of batch, until before last one, on all dimensions
        loss_neg_sampling, loss_classif_domain, loss_desc, loss_title, loss_rk, loss_classif_exp = 0, 0, 0, 0, 0, 0
        for num, job_emb in enumerate(jobs_emb[:, :-1, :]):
            domain_job_rep, experience_job_rep, predicted_domain_ind, predicted_desc, predicted_exp_ind, predicted_title = self.forward(job_emb, num)
            # domain loss
            loss_neg_sampling += torch.nn.functional.triplet_margin_with_distance_loss(domain_job_rep,
                                                                                      self.domain_reps[industry],
                                                                                      self.sample_neg_examples(industry, self.hp.neg_examples))
            loss_classif_domain += torch.nn.functional.cross_entropy(predicted_domain_ind, industry)
            loss_desc += torch.nn.functional.cross_entropy(predicted_desc, jobs_descriptions[:, num, :])

            # experience loss
            prev_job_rep = self.experience_encoder.forward(jobs_emb[:, num + 1, :])
            loss_rk += torch.nn.functional.margin_ranking_loss(experience_job_rep, prev_job_rep,
                                                              torch.ones(self.hp.b_size))
            loss_classif_exp += - torch.nn.functional.cross_entropy(predicted_exp_ind, industry)
            loss_title += torch.nn.functional.cross_entropy(predicted_title, jobs_titles[:, num, :])

        domain_loss = (self.coeff[0] * loss_neg_sampling / num +
                       self.coeff[1] * loss_classif_domain / num +
                       self.coeff[2] * loss_desc / num) / sum(self.coeff[:3])
        experience_loss = (self.coeff[3] * loss_rk / num +
                           self.coeff[4] * loss_classif_exp / num +
                           self.coeff[5] * loss_title / num) / sum(self.coeff[3:])

        self.log_losses_valid(loss_neg_sampling, loss_classif_domain, loss_desc, loss_rk, loss_classif_exp,
                              loss_title, experience_loss, domain_loss)
        return {"valid_exp_loss": experience_loss, "valid_domain_loss": domain_loss}

    def configure_optimizers(self):
        pass

    def sample_neg_examples(self, industry, n):
        ipdb.set_trace()


    def log_losses_train(self, *args):
        ipdb.set_trace()
        for loss in args[:-2]:
            self.log(str(loss.__name__.split("_")[1:]), loss, on_step=True, on_epoch=False)
        self.log("loss_experience", args[-2], on_step=True, on_epoch=False)
        self.log("loss_demain", args[-1], on_step=True, on_epoch=False)
