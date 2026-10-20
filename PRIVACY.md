# Privacy Policy — "Ask OEKG" Chatbot

_Last updated: 2026-10-12_

This policy explains how the **Ask OEKG** chatbot processes personal data. It
covers the chatbot application itself. For the Open Energy Platform website,
user accounts and the web server that hosts this application, the
[OEP Privacy Policy](https://openenergyplatform.org/legal/privacy_policy/)
applies in addition.

---

## 1. Identity of the controller

The controller responsible for the processing described here is:

- **Name:** Till Mossakowski
- **Address:** Universitätsplatz 2, 39126 Magdeburg, Germany
- **E-mail:** <glauer@iks.cs.ovgu.de>

If a data protection officer has been appointed for the controller's
institution, their contact details can be obtained via the address above.

---

## 2. What data is processed, why, on what legal basis, and for how long

### 2.1 Your chat input (questions)

**Data:** the free text you type into the chat, plus the last few turns of the
current conversation used as context.

**Purpose:** to translate your question into a SPARQL query with a language
model, execute it against the Open Energy Knowledge Graph, and summarise the
result. The text of your current question is additionally converted into a
numeric vector ("embedding") to find the relevant parts of the ontology. Which
of this data goes to which service is set out in section 3.

**Legal basis:** your consent, Art. 6 (1) (a) GDPR. You give this consent via
the checkbox shown before the chatbot becomes usable, and you can withdraw it at
any time (see section 5).

**Storage period:**

- **In memory:** the conversation is held in the server-side session for the
  duration of your session only and is limited to the most recent user turns.
  It is discarded when the session ends, when you reload the page, or when you
  withdraw consent.
- **In the request log:** each request is written to a rotating log file
  (`logs/requests.log`) with a timestamp, the **text of your question**, and
  technical metadata (whether a cached query was used, whether an error
  occurred, the number of result rows, and the processing duration). The log
  does **not** contain IP addresses, account names or any other identifier
  linking an entry to you. Entries are removed automatically once the file
  rotates.
- **In the error log:** warnings and exceptions are written to
  `logs/errors.log` and may incidentally contain fragments of a question. The
  same rotation applies.

The request log additionally records the moment consent is given or withdrawn,
together with the version of this document — again without any identifier. This
serves the controller's obligation to demonstrate that consent was obtained
(Art. 7 (1) GDPR).

> ⚠️ **Please do not enter personal, confidential or otherwise sensitive
> information into the chat.** The chatbot is a research and exploration tool
> for public, openly licensed energy data. It has no need for personal data and
> you should not supply any.

### 2.2 The shared question cache

**Data:** the text of a question, the SPARQL query generated for it, the time it
was created and last used, a usage counter, and a numeric embedding
(vector representation) of the question, computed as described in section 3.1.

**Purpose:** to let any user re-run a previously answered question instantly,
without the language model having to generate a new SPARQL query. The result is
still summarised by the language model as described in section 3.1.

**Important:** this cache is **shared between all users** of this instance. Its
entries are listed in the "Saved questions" sidebar and are therefore visible to
everyone using the chatbot. Nothing is written to it automatically — an entry is
only created when a user **explicitly** clicks "Cache this question" or adds one
through the manual form. Do not save a question that contains information you do
not want to publish.

**Legal basis:** your consent, Art. 6 (1) (a) GDPR, given by that explicit
action.

**Storage period:** until the entry is removed by an operator (the cache can be
cleared from the sidebar). Usage counters and the "last used" timestamp are
updated on every hit.

### 2.3 Technical access data

Requests to the web server (IP address, time, requested resource, browser
identification) are processed by the hosting infrastructure. This is described
in the [OEP Privacy Policy](https://openenergyplatform.org/legal/privacy_policy/),
which states a retention period of 14 days for server logs. The chatbot itself
does not read, store or evaluate your IP address.

The application does not set any tracking cookies of its own, and the
usage-statistics telemetry of the underlying Streamlit framework is switched
off in the application's configuration. Only the technically necessary session
cookie of the web framework is used, which is deleted when your session ends.

---

## 3. Recipients and transfers to third countries

### 3.1 Osnabrück University (language model and embeddings)

Two services are involved, both operated by

**Universität Osnabrück**, Neuer Graben 29 / Schloss, 49074 Osnabrück, Germany:

- **KiWi** (`kiwi-secure.uni-osnabrueck.de`) runs the **language model**. To
  generate a SPARQL query, to summarise a result and to explain a query, your
  question, the last few turns of the current conversation, the relevant
  retrieved ontology excerpts and — where applicable — the retrieved result
  rows are transmitted there.
- The **AI gateway** (`litellm.uni-osnabrueck.de`) computes the **embedding**
  of your question: the numeric vector used to find the relevant parts of the
  ontology and to compare a new question against the shared question cache
  (section 2.2). Only the text of your current question is sent for this.

Both services are hosted in the **same university computing centre**, and both
models (Qwen3.5 with 122B parameters for the chat, BGE-M3 for the embeddings —
openly available models) run on Osnabrück University's own infrastructure. Your
input is therefore processed **within Germany**, is not passed on to any
commercial model provider, and is **not stored** by either service according to
their operator.

### 3.2 Open Energy Platform (knowledge graph and datasets)

Before the language model is called, names and terms taken **verbatim from
your question** (quoted phrases and capitalised words) are sent to the SPARQL
endpoint of the Open Energy Platform (`openenergyplatform.org`) to look them up
in the knowledge graph; if a query later returns no results, similar entries are
looked up once more in the same way. The generated SPARQL query — which may also
contain words from your question — is then executed there, as are any later
requests for dataset rows or metadata. All of these requests are authenticated
with the **application's own service token** — your identity is not transmitted,
and the platform receives only the queries themselves, not who asked them.

### 3.3 No other recipients

No data is sold, shared for advertising purposes, or passed to any other third
party. No web analytics, tracking or advertising services are used.

### 3.4 Transfers to third countries

Your input is **not transferred to a country outside the European Union**. Both
the language model and the embedding model run on Osnabrück University's
infrastructure in Germany (section 3.1), and the knowledge graph is queried at
the Open Energy Platform (section 3.2).

The chatbot is nevertheless made available only after your explicit consent
(Art. 6 (1) (a) GDPR), because your question is sent to an external language
model and its text is recorded in the request log (section 2.1).

---

## 4. Rights of the data subject

You have the right, in relation to the controller named in section 1, to:

- **information** about the personal data processed about you (Art. 15 GDPR);
- **rectification** of inaccurate data (Art. 16 GDPR);
- **erasure** (Art. 17 GDPR);
- **restriction of processing** (Art. 18 GDPR);
- **data portability** (Art. 20 GDPR);
- **object** to processing (Art. 21 GDPR).

Please note that requests concerning chat input can only be fulfilled where the
data can still be identified. Because the request log deliberately contains no
identifier linking an entry to a person, it is generally **not possible** for
the controller to attribute a logged question to you (Art. 11 GDPR). If you wish
to have a specific entry of the shared question cache removed, please state the
question text in your request.

You also have the right to **lodge a complaint with a supervisory authority**
(Art. 77 GDPR). For the controller named above, this is the
Landesbeauftragter für den Datenschutz Sachsen-Anhalt.

---

## 5. Right to object and withdrawal of consent

Processing of your chat input is based on your consent. You may **withdraw that
consent at any time and with effect for the future**, without giving reasons and
without any disadvantage.

To withdraw it, use the **"Withdraw consent"** button in the chatbot's sidebar.
This ends the processing immediately, clears the current conversation from the
session, and returns you to the consent screen. Withdrawing does not affect the
lawfulness of processing carried out before the withdrawal.

To have entries removed from the shared question cache, or to exercise any other
right, contact the controller at the address in section 1.

---

## 6. Obligation to provide personal data

You are under **no obligation** to provide any personal data. Using the chatbot
does not require an account, and no field asks for personal data. If you decline
consent, the chatbot is simply not available to you; the underlying data remains
freely accessible on [openenergyplatform.org](https://openenergyplatform.org/).

---

## 7. Automated decision-making and profiling

The chatbot generates answers automatically using a language model. This
processing is **not** automated decision-making within the meaning of
Art. 22 GDPR: it produces no legal effect concerning you and does not similarly
significantly affect you. No profiles of users are created, and no behaviour is
evaluated across sessions.

Please note that the answers are generated by a statistical model and **may be
incomplete or incorrect**. They are not a substitute for the authoritative data
published on the Open Energy Platform.

---

## 8. Changes to this policy

This policy may be updated when the chatbot's processing changes. The
application detects changes to this document automatically and asks for renewed
consent when its content has been modified.
