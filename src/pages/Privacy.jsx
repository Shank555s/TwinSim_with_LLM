import { SITE } from '../site.js'
import { Card, PageHead } from '../ui.jsx'

export default function Privacy() {
  return (<>
    <PageHead title="Privacy policy" sub={`Last updated ${SITE.lastUpdated}.`} />
    <Card className="legal">
      <p>
        {SITE.name} is a research tool. It runs a small language model on the server that hosts this site and stores
        everything in a database on that same server. This page describes what is stored and who can see it.
      </p>

      <h2>What is stored</h2>
      <ul>
        <li>Personas and scenarios that anyone creates on this site.</li>
        <li>Every generated answer, its fidelity score, the generation settings (model, seed, temperature) and any human rating.</li>
        <li>Training runs, model checkpoints and a history log of actions taken in the app.</li>
      </ul>
      <p>
        There are no user accounts. The site does not ask for your name, email address or any other personal details,
        and you should not enter personal information about real people into a persona.
      </p>

      <h2>Who can see it</h2>
      <p>
        Because there are no accounts, anything saved on this site (personas, answers, ratings, history) is visible to
        anyone else who opens the same site. Treat it as a shared lab notebook, not a private space.
      </p>

      <h2>What is not used</h2>
      <ul>
        <li>No analytics, advertising or tracking scripts.</li>
        <li>No cookies.</li>
        <li>No third-party AI service. Answers come from the model stored on this server.</li>
        <li>Fonts and icons are served from this site, so your browser does not contact any other domain.</li>
      </ul>

      <h2>Stored in your browser</h2>
      <p>
        Three interface preferences are kept in your browser's local storage: whether the sidebar is collapsed,
        whether animations are reduced, and when you last opened notifications. They never leave your browser and
        you can clear them in your browser settings.
      </p>

      <h2>Server logs</h2>
      <p>
        The web server records each request (IP address, time and page) in its log, as most web servers do. These
        logs are used only to keep the site running.
      </p>

      <h2>Deleting data</h2>
      <p>
        Archiving a persona hides it in the app but keeps the record so past experiments stay reproducible. To have
        data removed permanently, contact the site owner{SITE.contactEmail ? ' below' : ''}.
      </p>

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
