import { SITE } from '../site.js'
import { Card, PageHead } from '../ui.jsx'

export default function Terms() {
  return (<>
    <PageHead title="Terms of use" sub={`Last updated ${SITE.lastUpdated}.`} />
    <Card className="legal">
      <p>By using {SITE.name} you agree to the following.</p>

      <h2>What this is</h2>
      <p>
        {SITE.name} is an academic research project. It is provided as is, without any warranty, and may change,
        reset its data or go offline at any time.
      </p>

      <h2>About the answers</h2>
      <ul>
        <li>Answers are produced by a small language model trained on synthetic, template-generated text.</li>
        <li>They do not describe or represent the views of any real person, including people who resemble a persona.</li>
        <li>Fidelity scores come from a rule-based scorer. They measure agreement with that rubric, not truth or quality.</li>
        <li>Do not use the answers or scores to make decisions about real people.</li>
      </ul>

      <h2>Acceptable use</h2>
      <ul>
        <li>Do not enter personal, sensitive or identifying information about real people.</li>
        <li>Do not enter content that is illegal, hateful or meant to harass anyone.</li>
        <li>Do not try to overload the server, for example by starting training jobs repeatedly.</li>
      </ul>
      <p>The site owner may remove content or restrict access to keep the service working.</p>

      <h2>Your content</h2>
      <p>
        Personas, scenarios and ratings you add are stored on the server and are visible to other visitors
        (see the Privacy policy). Do not add anything you are not allowed to share.
      </p>

      <h2>Code</h2>
      <p>No open-source licence has been applied to this project's code yet.</p>

      {(SITE.owner || SITE.contactEmail) && (<>
        <h2>Contact</h2>
        <p>
          {SITE.owner && <>{SITE.owner}<br /></>}
          {SITE.contactEmail && <a href={`mailto:${SITE.contactEmail}`}>{SITE.contactEmail}</a>}
        </p>
      </>)}
    </Card>
  </>)
}
