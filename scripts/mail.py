#!/usr/bin/python

import smtplib
import os
import yaml
import utils
from email.mime.text import MIMEText


class Postman():

    """A class that is used to send mails to a user."""

    def __init__(self):

        self.success = self.load_credentials()
        self.smtpObj = smtplib.SMTP(self.credentials["server_name"],
                                    self.credentials["smtp_port"])
        self.smtpObj.ehlo()
        self.smtpObj.starttls()
        self.smtpObj.login(self.credentials["user_name"],
                           self.credentials["password"])

    def load_credentials(self):
        credentials_file = os.getenv("CREDENTIALS_FILE")
        if not credentials_file:
            credentials_file = "credentials.yml"

        completion = False
        try:
            doc = open(credentials_file, "r")
            self.credentials = yaml.safe_load(doc)
            completion = True
        except:
            print (utils.BRED + "Could not open the credentials file" +
                   utils.ENDC)
            completion = False

        return completion

    def send_mail(self, receivers, msg):
        if not self.success:
            return None
        try:
            # if isinstance(receivers, list):
                # receivers = "".join(receivers)
            body = ""
            for key, value in msg.iteritems():
                body += key + " : " + str(value) + " \n"
            # body = "" + message + ""
            message = MIMEText(body)
            message["Subject"] = "Training Result"
            message["From"] = self.credentials["user_name"] 
            message["To"] = ", ".join(receivers)

            self.smtpObj.sendmail(self.credentials["user_name"], receivers,
                                  message.as_string())
            print utils.BGREEN + "Successfully sent email" + utils.ENDC
        except:
            print utils.BRED + "Error: unable to send email" + utils.ENDC
